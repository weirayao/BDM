from __future__ import annotations

"""Loss computation utilities for BD3LM Flax training.

This implements the *SUBS* parameterisation of Block Diffusion that the
original PyTorch code uses by default.  Other variants (SEDD, AR) can be
ported later but are not strictly necessary for pre-training on OWT.
"""

from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from .bd3lm_arch import BD3LMConfig, BD3LMFlax
from .noise_schedule import NoiseSchedule


# Use the metrics dictionary in a JAX-traceable way
# Don't attempt to convert to Python types in the compiled code path
def create_metrics(loss_value):
    """Create a simple dictionary of metrics that stays as JAX arrays."""
    return {"loss": loss_value}  # Keep as JAX array


# ---------------------------------------------------------------------------
# Helper functions                                                           
# ---------------------------------------------------------------------------


def _sample_t(batch_size: int, seqlen: int, rng: jax.random.KeyArray, eps_min: float, eps_max: float):
    """Draw per-token diffusion times as in original implementation."""
    num_tokens = seqlen
    # Sample one scalar per *block* then repeat ‑ simplifies implementation.
    t = jax.random.uniform(rng, (batch_size, num_tokens), minval=eps_min, maxval=eps_max)
    return t.astype(jnp.float32)


def _sigma_from_p(p: jnp.ndarray, schedule: NoiseSchedule):
    """sigma = -log(1-p) clipped by schedule.sigma_max (LogLinear only)."""
    # schedule must be LogLinearNoise for this helper to be useful.
    return jnp.minimum(-jnp.log1p(-p), schedule.sigma_max)


def q_xt(mask_index: int, x: jnp.ndarray, p: jnp.ndarray, rng: jax.random.KeyArray):
    """Apply binary mask with probability p (per token)."""
    bern = jax.random.bernoulli(rng, p=p)
    xt = jnp.where(bern, mask_index, x)
    return xt


# ---------------------------------------------------------------------------
# Top-level loss
# ---------------------------------------------------------------------------

def diffusion_loss(
    cfg: BD3LMConfig,
    model: BD3LMFlax,
    params: Any,
    schedule: NoiseSchedule,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.KeyArray,
) -> Tuple[jnp.ndarray, Dict[str, float], Any]:
    """Compute SUBS loss for one batch.

    Returns
    -------
    loss : scalar – mean NLL over tokens (masked by attention_mask)
    metrics : dict – for logging
    new_mutables : any – to pass back to TrainState
    """
    input_ids = batch["input_ids"]  # (B, S)
    attn_mask = batch["attention_mask"]  # (B, S)
    B, S = input_ids.shape

    # 1. Sample t and compute schedule quantities
    rng, t_rng, mask_rng = jax.random.split(rng, 3)
    t = _sample_t(B, S, t_rng, cfg.sampling_eps_min, cfg.sampling_eps_max)
    loss_scale, p = schedule(t)

    # 2. Corrupt tokens (x_t) - use vocab_size from cfg instead of accessing model.backbone
    mask_index = cfg.vocab_size - 1  # Last token is mask token
    xt = q_xt(mask_index, input_ids, p, mask_rng)

    # 3. Handle cross-attention variant (concat x_t + x0)
    if cfg.cross_attn:
        model_input = jnp.concatenate([xt, input_ids], axis=1)
    else:
        model_input = xt

    # 4. Forward pass – returns logits for all positions
    timesteps = _sigma_from_p(p[:, 0:1], schedule)  # (B, 1)
    
    # Create a dedicated PRNG key for the dropout
    rng, dropout_rng = jax.random.split(rng)
    
    # Flax model.apply with mutable=... returns (outputs, mutated_variables)
    output, new_mutables = model.apply(
        {"params": params},
        model_input,
        timesteps=timesteps,
        mutable=["batch_stats"] if cfg.adaln else [],
        deterministic=False,
        rngs={"dropout": dropout_rng},  # Provide dropout PRNG key
    )
    
    # Extract logits from the model output
    logits = output

    if cfg.cross_attn:
        logits = logits[:, :S]  # keep x_t stream only for loss

    # 5. SUBS parametrisation – see original Diffusion._subs_parameterization.
    neg_inf = -1e8
    
    # Update logits for mask token (adding neg_inf to mask index)
    logits = logits.at[:, :, mask_index].add(neg_inf)
    
    # Normalize logits (subtract logsumexp for numerical stability)
    logits = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    
    # Create boolean mask for unmasked tokens
    unmasked = (xt != mask_index)
    
    # Instead of using boolean indexing directly, use arithmetic operations with the mask
    # to selectively update values (JAX-compatible approach)
    unmasked_mask = unmasked[:, :, None]  # Add vocab dimension for broadcasting
    target_mask = jnp.zeros_like(logits)
    
    # For unmasked tokens, set all logits to neg_inf except the true token
    neg_inf_mask = jnp.where(unmasked_mask, jnp.ones_like(logits) * neg_inf, jnp.zeros_like(logits))
    logits = logits + neg_inf_mask
    
    # For unmasked tokens, set the true token logit to 0
    # Create one-hot encoding for the target tokens
    target_one_hot = jax.nn.one_hot(input_ids, logits.shape[-1])
    # Only apply for unmasked tokens
    zeros_for_targets = jnp.where(unmasked_mask, target_one_hot * -neg_inf, jnp.zeros_like(logits))
    logits = logits + zeros_for_targets

    # 6. NLL
    log_p_theta = jnp.take_along_axis(logits, input_ids[..., None], axis=-1)[..., 0]
    per_token_loss = -loss_scale * log_p_theta * attn_mask
    mean_loss = per_token_loss.sum() / attn_mask.sum()

    metrics = create_metrics(mean_loss)

    # Ensure we return exactly (loss, metrics, new_mutables)
    return mean_loss, metrics, new_mutables 