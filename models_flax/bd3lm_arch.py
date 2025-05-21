################################################################################
# BSD 3-Clause License (same as original project)
#
#  BD3LM Flax implementation
#  -------------------------------------------------------------
#  This file contains a minimal yet fully-functional Flax / JAX
#  re-write of the PyTorch DIT-BD3LM model that is used for
#  Block Diffusion language modelling.  The goal is *feature
#  parity for training*; sampling-time KV-cache and FlexAttention
#  kernels can be added later.
#
#  Author: migration script (o3-assistant)
################################################################################

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import flax.struct  # immutable dataclass

###############################################################################
# Configuration dataclass                                                    #
###############################################################################


@flax.struct.dataclass
class BD3LMConfig:
    """Lightweight config object (pytree) usable with JAX / Flax."""
    # Architecture
    block_size: int = 2
    vocab_size: int = 50_258
    model_length: int = 1_024  # context length for the diffusion input
    cross_attn: bool = True  # use two-stream (x_t + x_0) attention mask
    adaln: bool = True  # AdaLayerNorm-Zero
    attn_backend: str = "sdpa"  # "sdpa" for now – FlexAttention may follow
    causal: bool = False  # Allow autoregressive cross-stream causal mask

    # Hidden dims
    hidden_dim: int = 768
    cond_dim: int = 128 + 1  # 129 in original paper
    n_blocks: int = 12
    n_heads: int = 12
    dropout: float = 0.1

    # Time-conditioning / diffusion
    time_conditioning: bool = False
    var_min: bool = True
    sampling_eps_min: float = 1e-3
    sampling_eps_max: float = 0.999

    # Precision settings
    dtype: Any = jnp.bfloat16  # pretty much required on TPU


###############################################################################
# Utility functions                                                          #
###############################################################################

def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Helper for rotary embeddings."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary(qkv: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embeddings to *packed* qkv tensor (… seq 3 heads dim)."""
    return (qkv * cos) + (rotate_half(qkv) * sin)


def block_diff_mask(
    q_idx: jnp.ndarray,
    kv_idx: jnp.ndarray,
    *,
    n: int,
    block_size: int,
) -> jnp.ndarray:
    """Vectorised version of the PyTorch mask – returns bool[Q, K]."""
    x0_flag_q = q_idx >= n
    x0_flag_k = kv_idx >= n

    block_q = jnp.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_k = jnp.where(x0_flag_k, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diag = (block_q == block_k) & (x0_flag_q == x0_flag_k)
    offset_block_causal = (block_q > block_k) & x0_flag_k & (~x0_flag_q)
    block_causal = (block_q >= block_k) & x0_flag_k & x0_flag_q
    breakpoint()
    return block_diag | offset_block_causal | block_causal


def make_attention_bias(mask: jnp.ndarray, dtype=jnp.float32) -> jnp.ndarray:
    """Convert boolean mask to large negative bias suitable for attention."""
    return jnp.where(mask, 0.0, jnp.finfo(dtype).min)

###############################################################################
# Core modules                                                               #
###############################################################################


class RotaryEmbedding(nn.Module):
    dim: int
    base: int = 10_000

    @nn.compact
    def __call__(self, seq_len: int, dtype=jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=dtype) / self.dim))
        t = jnp.arange(seq_len, dtype=dtype)
        freqs = jnp.einsum("n,d->nd", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)[None, :, None, None, :]  # placeholder shape [1, seq, 1, 1, dim]
        sin = jnp.sin(emb)[None, :, None, None, :]
        # replicate to 3 (qkv) axis
        cos = jnp.tile(cos, (1, 1, 3, 1, 1))  # [1, seq, 3, 1, dim]
        sin = jnp.tile(sin, (1, 1, 3, 1, 1))
        # for v vectors: identity transformation
        cos = cos.at[:, :, 2, :, :].set(1.0)
        sin = sin.at[:, :, 2, :, :].set(0.0)
        return cos, sin


class TimestepEmbedder(nn.Module):
    hidden_size: int
    freq_dim: int = 256

    @staticmethod
    def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10_000) -> jnp.ndarray:
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(half, dtype=t.dtype) / half
        )
        args = t[..., None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        emb = self.timestep_embedding(timesteps, self.freq_dim).astype(jnp.float32)
        x = nn.Dense(self.hidden_size)(emb)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size)(x)
        return x


class MLP(nn.Module):
    hidden_dim: int
    dropout: float
    def setup(self):
        self.fc1 = nn.Dense(self.hidden_dim * 4, use_bias=True)
        self.fc2 = nn.Dense(self.hidden_dim, use_bias=True)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, x, deterministic):
        y = self.fc1(x)
        y = nn.gelu(y, approximate="tanh")
        y = self.dropout_layer(y, deterministic=deterministic)
        y = self.fc2(y)
        return y


class DDiTBlock(nn.Module):
    cfg: BD3LMConfig

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.qkv_proj = nn.Dense(3 * self.cfg.hidden_dim, use_bias=False)
        self.out_proj = nn.Dense(self.cfg.hidden_dim, use_bias=False)
        self.mlp = MLP(self.cfg.hidden_dim, self.cfg.dropout)
        if self.cfg.adaln:
            self.ada_mod = nn.Dense(6 * self.cfg.hidden_dim, use_bias=True, kernel_init=jax.nn.initializers.zeros)
        self.dropout_layer = nn.Dropout(rate=self.cfg.dropout)
        self.rotary = RotaryEmbedding(self.cfg.hidden_dim // self.cfg.n_heads)

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        sigma_emb: Optional[jnp.ndarray],
        attn_mask: Optional[jnp.ndarray],
        deterministic: bool,
        custom_attn_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        assert x.ndim == 3  # (B, S, D)
        B, S, D = x.shape
        H = self.cfg.n_heads
        head_dim = D // H

        # Use custom attention mask if provided (for sampling)
        if custom_attn_mask is not None:
            attn_mask = custom_attn_mask
            
        # AdaLN parameters
        if self.cfg.adaln:
            assert sigma_emb is not None, "AdaLN enabled but sigma_emb is None"
            # Calculate output dimension to ensure proper reshaping
            hidden_dim = self.cfg.hidden_dim
            # Compute ada_mod outputs and reshape properly
            ada_mod_output = self.ada_mod(sigma_emb)  # [B, 6*hidden_dim]
            ada_mod_output = ada_mod_output.reshape(-1, 6, hidden_dim)  # [B, 6, hidden_dim]
            
            # Extract the components directly - no splitting needed
            shift_msa = ada_mod_output[:, 0][:, None, :]  # [B, 1, hidden_dim]
            scale_msa = ada_mod_output[:, 1][:, None, :]
            gate_msa = ada_mod_output[:, 2][:, None, :]
            shift_mlp = ada_mod_output[:, 3][:, None, :]
            scale_mlp = ada_mod_output[:, 4][:, None, :]
            gate_mlp = ada_mod_output[:, 5][:, None, :]
        else:
            shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None

        # === MSA ===
        y = self.norm1(x)
        if self.cfg.adaln:
            y = y * (1 + scale_msa) + shift_msa
        qkv = self.qkv_proj(y)
        qkv = qkv.reshape(B, S, 3, H, head_dim)
        cos, sin = self.rotary(S, dtype=x.dtype)
        qkv = apply_rotary(qkv, cos, sin)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
        # (B, H, S, D_h)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask  # mask is broadcastable
        attn_probs = nn.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout_layer(attn_probs, deterministic=deterministic)
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout_layer(attn_out, deterministic=deterministic)

        if self.cfg.adaln:
            x = x + gate_msa * attn_out
        else:
            x = x + attn_out

        # === MLP ===
        y = self.norm2(x)
        if self.cfg.adaln:
            y = y * (1 + scale_mlp) + shift_mlp
        y = self.mlp(y, deterministic=deterministic)
        y = self.dropout_layer(y, deterministic=deterministic)
        if self.cfg.adaln:
            x = x + gate_mlp * y
        else:
            x = x + y
        return x


class BD3LMBackbone(nn.Module):
    cfg: BD3LMConfig

    def setup(self):
        self.token_embed = nn.Embed(self.cfg.vocab_size, self.cfg.hidden_dim)
        if self.cfg.adaln:
            self.sigma_embed = TimestepEmbedder(self.cfg.cond_dim)
        self.blocks = [DDiTBlock(self.cfg) for _ in range(self.cfg.n_blocks)]
        self.norm_final = nn.LayerNorm()
        self.classifier = nn.Dense(self.cfg.vocab_size)

        if self.cfg.cross_attn:
            # Pre-compute full attention mask for training (2n x 2n)
            mask = block_diff_mask(
                jnp.arange(self.cfg.model_length * 2)[:, None],
                jnp.arange(self.cfg.model_length * 2)[None, :],
                n=self.cfg.model_length,
                block_size=self.cfg.block_size,
            )
            self.register("attn_mask", make_attention_bias(mask, dtype=self.cfg.dtype))

    def register(self, name: str, value):
        self.param(name, lambda *_: value)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        *,
        timesteps: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_hidden_states: bool = False,
        custom_attn_mask: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]]]:
        """Forward pass returning (logits, hidden_states?)."""
        x = self.token_embed(input_ids)
        hidden_states = [x] if output_hidden_states else None

        if self.cfg.adaln:
            assert timesteps is not None, "timesteps required when AdaLN is enabled"
            sigma_emb = nn.silu(self.sigma_embed(timesteps))  # (B, cond_dim)
        else:
            sigma_emb = None

        # Build mask once for given sequence length if cross_attn disabled
        if custom_attn_mask is not None:
            # Use the custom mask if provided (for sampling)
            attn_mask = custom_attn_mask
        elif self.cfg.cross_attn:
            attn_mask = self.variables["params"]["attn_mask"]
            # For training: slice since seq_len may be shorter (e.g., cropped)
            L = input_ids.shape[1]
            attn_mask = attn_mask[:L, :L]
            # broadcast to heads / batch later
            attn_mask = attn_mask[None, None, :, :]
        else:
            attn_mask = None

        for block in self.blocks:
            x = block(
                x,
                sigma_emb=sigma_emb,
                attn_mask=attn_mask,
                deterministic=deterministic,
                custom_attn_mask=custom_attn_mask,
            )
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm_final(x)
        logits = self.classifier(x)
        return (logits, hidden_states) if output_hidden_states else (logits, None)


###############################################################################
# Public model wrapper compatible with training script                       #
###############################################################################


class BD3LMFlax(nn.Module):
    cfg: BD3LMConfig

    def setup(self):
        self.backbone = BD3LMBackbone(self.cfg)
        if self.cfg.var_min:
            self.sampling_eps_min = self.cfg.sampling_eps_min
            self.sampling_eps_max = self.cfg.sampling_eps_max

    # quick alias so training code can call model.apply(...).logits etc.
    def __call__(self, *args, **kwargs):
        logits, hidden_states = self.backbone(*args, **kwargs)
        return logits 