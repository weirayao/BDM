from __future__ import annotations

"""Utility functions for BD3LM JAX/Flax implementation.

This file provides helper functions for integrating the Flax implementation
with the PyTorch code, including model conversion utilities.
"""

import os
from typing import Dict, Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import serialization
from flax.traverse_util import flatten_dict, unflatten_dict

from .bd3lm_arch import BD3LMConfig, BD3LMFlax


def save_checkpoint(params: Dict, path: str, overwrite: bool = False) -> None:
    """Save a Flax model checkpoint."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))


def load_checkpoint(path: str) -> Dict:
    """Load a Flax model checkpoint."""
    with open(path, 'rb') as f:
        params = serialization.from_bytes(None, f.read())
    return params


def convert_pytorch_state_dict_to_flax(
    pt_state_dict: Dict[str, torch.Tensor],
    flax_model: BD3LMFlax,
    config: BD3LMConfig
) -> Dict:
    """Convert a PyTorch state dict to a Flax parameter dict.
    
    This handles the differences in parameter naming and structure
    between the PyTorch and Flax implementations.
    """
    # Initialize the Flax model to get the expected parameter structure
    dummy_input = jnp.ones((1, config.model_length), dtype=jnp.int32)
    dummy_timesteps = jnp.ones((1, 1), dtype=jnp.float32)
    params = flax_model.init(jax.random.PRNGKey(0), dummy_input, timesteps=dummy_timesteps)['params']
    
    # Convert PyTorch state dict to flat dict
    pt_flat_dict = {}
    for k, v in pt_state_dict.items():
        if k.startswith('backbone.'):
            k = k[len('backbone.'):]
        pt_flat_dict[k] = jnp.array(v.detach().cpu().numpy())
    
    # Convert Flax params to flat dict
    flax_flat_dict = flatten_dict(params)
    
    # Mapping between PyTorch and Flax parameter names
    name_mapping = {
        # Embedding layers
        'token_embed.weight': ('backbone', 'token_embed', 'embedding'),
        
        # Final norm and classifier
        'norm_final.weight': ('backbone', 'norm_final', 'scale'),
        'norm_final.bias': ('backbone', 'norm_final', 'bias'),
        'classifier.weight': ('backbone', 'classifier', 'kernel'),
        'classifier.bias': ('backbone', 'classifier', 'bias'),
    }
    
    # Add block mappings
    for i in range(config.n_blocks):
        # Layer norms
        name_mapping[f'blocks.{i}.norm1.weight'] = ('backbone', 'blocks_{i}', 'norm1', 'scale')
        name_mapping[f'blocks.{i}.norm1.bias'] = ('backbone', 'blocks_{i}', 'norm1', 'bias')
        name_mapping[f'blocks.{i}.norm2.weight'] = ('backbone', 'blocks_{i}', 'norm2', 'scale')
        name_mapping[f'blocks.{i}.norm2.bias'] = ('backbone', 'blocks_{i}', 'norm2', 'bias')
        
        # Attention weights
        name_mapping[f'blocks.{i}.qkv_proj.weight'] = ('backbone', 'blocks_{i}', 'qkv_proj', 'kernel')
        name_mapping[f'blocks.{i}.out_proj.weight'] = ('backbone', 'blocks_{i}', 'out_proj', 'kernel')
        
        # MLP
        name_mapping[f'blocks.{i}.mlp.fc1.weight'] = ('backbone', 'blocks_{i}', 'mlp', 'fc1', 'kernel')
        name_mapping[f'blocks.{i}.mlp.fc1.bias'] = ('backbone', 'blocks_{i}', 'mlp', 'fc1', 'bias')
        name_mapping[f'blocks.{i}.mlp.fc2.weight'] = ('backbone', 'blocks_{i}', 'mlp', 'fc2', 'kernel')
        name_mapping[f'blocks.{i}.mlp.fc2.bias'] = ('backbone', 'blocks_{i}', 'mlp', 'fc2', 'bias')
        
        # AdaLN
        if config.adaln:
            name_mapping[f'blocks.{i}.ada_mod.weight'] = ('backbone', 'blocks_{i}', 'ada_mod', 'kernel')
            name_mapping[f'blocks.{i}.ada_mod.bias'] = ('backbone', 'blocks_{i}', 'ada_mod', 'bias')
    
    # Sigma embedder (if using AdaLN)
    if config.adaln:
        name_mapping['sigma_embed.0.weight'] = ('backbone', 'sigma_embed', 'Dense_0', 'kernel')
        name_mapping['sigma_embed.0.bias'] = ('backbone', 'sigma_embed', 'Dense_0', 'bias')
        name_mapping['sigma_embed.2.weight'] = ('backbone', 'sigma_embed', 'Dense_1', 'kernel')
        name_mapping['sigma_embed.2.bias'] = ('backbone', 'sigma_embed', 'Dense_1', 'bias')
    
    # Apply the mapping
    new_params = {}
    for pt_name, flax_path in name_mapping.items():
        if pt_name in pt_flat_dict:
            # Handle block index formatting
            formatted_path = []
            for part in flax_path:
                if part.startswith('blocks_'):
                    formatted_path.append(f'blocks_{i}')
                else:
                    formatted_path.append(part)
            
            flax_key = tuple(formatted_path)
            if flax_key in flax_flat_dict:
                # Get PyTorch tensor and convert to JAX array
                pt_tensor = pt_flat_dict[pt_name]
                
                # For dense layer kernels, transpose weight matrices
                if 'kernel' in flax_key[-1] and 'embedding' not in flax_key:
                    pt_tensor = pt_tensor.T
                
                # Update the parameter
                flax_flat_dict[flax_key] = pt_tensor
    
    # Rebuild the nested structure
    return {'params': unflatten_dict(flax_flat_dict)}


def convert_flax_state_dict_to_pytorch(
    flax_params: Dict, 
    pt_model: Any,
    config: BD3LMConfig
) -> Dict[str, torch.Tensor]:
    """Convert a Flax parameter dict to a PyTorch state dict.
    
    This handles the differences in parameter naming and structure
    between the Flax and PyTorch implementations.
    """
    # Extract just the params
    if 'params' in flax_params:
        flax_params = flax_params['params']
    
    # Flatten Flax params
    flax_flat_dict = flatten_dict(flax_params)
    
    # Create new PyTorch state dict
    pt_state_dict = {}
    
    # Mapping between Flax and PyTorch parameter names (inverse of the above)
    name_mapping = {
        ('backbone', 'token_embed', 'embedding'): 'backbone.token_embed.weight',
        ('backbone', 'norm_final', 'scale'): 'backbone.norm_final.weight',
        ('backbone', 'norm_final', 'bias'): 'backbone.norm_final.bias',
        ('backbone', 'classifier', 'kernel'): 'backbone.classifier.weight',
        ('backbone', 'classifier', 'bias'): 'backbone.classifier.bias',
    }
    
    # Add block mappings
    for i in range(config.n_blocks):
        # Layer norms
        name_mapping[('backbone', f'blocks_{i}', 'norm1', 'scale')] = f'backbone.blocks.{i}.norm1.weight'
        name_mapping[('backbone', f'blocks_{i}', 'norm1', 'bias')] = f'backbone.blocks.{i}.norm1.bias'
        name_mapping[('backbone', f'blocks_{i}', 'norm2', 'scale')] = f'backbone.blocks.{i}.norm2.weight'
        name_mapping[('backbone', f'blocks_{i}', 'norm2', 'bias')] = f'backbone.blocks.{i}.norm2.bias'
        
        # Attention weights
        name_mapping[('backbone', f'blocks_{i}', 'qkv_proj', 'kernel')] = f'backbone.blocks.{i}.qkv_proj.weight'
        name_mapping[('backbone', f'blocks_{i}', 'out_proj', 'kernel')] = f'backbone.blocks.{i}.out_proj.weight'
        
        # MLP
        name_mapping[('backbone', f'blocks_{i}', 'mlp', 'fc1', 'kernel')] = f'backbone.blocks.{i}.mlp.fc1.weight'
        name_mapping[('backbone', f'blocks_{i}', 'mlp', 'fc1', 'bias')] = f'backbone.blocks.{i}.mlp.fc1.bias'
        name_mapping[('backbone', f'blocks_{i}', 'mlp', 'fc2', 'kernel')] = f'backbone.blocks.{i}.mlp.fc2.weight'
        name_mapping[('backbone', f'blocks_{i}', 'mlp', 'fc2', 'bias')] = f'backbone.blocks.{i}.mlp.fc2.bias'
        
        # AdaLN
        if config.adaln:
            name_mapping[('backbone', f'blocks_{i}', 'ada_mod', 'kernel')] = f'backbone.blocks.{i}.ada_mod.weight'
            name_mapping[('backbone', f'blocks_{i}', 'ada_mod', 'bias')] = f'backbone.blocks.{i}.ada_mod.bias'
    
    # Sigma embedder (if using AdaLN)
    if config.adaln:
        name_mapping[('backbone', 'sigma_embed', 'Dense_0', 'kernel')] = 'backbone.sigma_embed.0.weight'
        name_mapping[('backbone', 'sigma_embed', 'Dense_0', 'bias')] = 'backbone.sigma_embed.0.bias'
        name_mapping[('backbone', 'sigma_embed', 'Dense_1', 'kernel')] = 'backbone.sigma_embed.2.weight'
        name_mapping[('backbone', 'sigma_embed', 'Dense_1', 'bias')] = 'backbone.sigma_embed.2.bias'
    
    # Apply the mapping
    for flax_key, pt_name in name_mapping.items():
        if flax_key in flax_flat_dict:
            # Get JAX array and convert to PyTorch tensor
            jax_array = flax_flat_dict[flax_key]
            
            # For dense layer kernels, transpose weight matrices
            if 'kernel' in flax_key[-1] and 'embedding' not in flax_key:
                jax_array = jax_array.T
            
            # Convert to PyTorch tensor
            pt_state_dict[pt_name] = torch.tensor(np.array(jax_array))
    
    return pt_state_dict


def convert_logits_to_probs(logits: jnp.ndarray, mask_index: int = None) -> jnp.ndarray:
    """Convert logits to probabilities, setting mask_index prob to 0 if provided."""
    if mask_index is not None:
        # Set mask token logit to a large negative value before softmax
        logits = logits.at[:, :, mask_index].set(-1e9)
    
    # Apply softmax to get probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    return probs


def get_nucleus_sampling_probs(probs: jnp.ndarray, p: float = 0.9) -> jnp.ndarray:
    """Apply nucleus sampling to probability distribution."""
    if p >= 1.0:
        return probs
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = jax.lax.sort_key_val(
        -probs, jnp.arange(probs.shape[-1])[None, None, :], dimension=-1)
    sorted_probs = -sorted_probs  # Restore positive values
    
    # Compute cumulative probabilities
    cumprobs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Create mask for probabilities within the nucleus
    nucleus_mask = cumprobs <= p
    
    # Always keep at least the top probability
    nucleus_mask = nucleus_mask.at[:, :, 0].set(True)
    
    # Apply the mask and normalize
    sorted_probs = sorted_probs * nucleus_mask
    
    # Use gather to restore the original order
    reordered_probs = jnp.zeros_like(probs)
    for i in range(probs.shape[-1]):
        idx = sorted_indices[:, :, i:i+1]
        value = sorted_probs[:, :, i:i+1]
        reordered_probs = reordered_probs.at[
            jnp.arange(probs.shape[0])[:, None, None],
            jnp.arange(probs.shape[1])[None, :, None],
            idx
        ].set(value)
    
    # Normalize
    reordered_probs = reordered_probs / (reordered_probs.sum(axis=-1, keepdims=True) + 1e-9)
    return reordered_probs 


def get_nucleus_sampling_probs_tpu_compatible(probs: jnp.ndarray, p: float = 0.9) -> jnp.ndarray:
    """TPU-compatible nucleus sampling with explicit handling of shape broadcasting."""
    if p >= 1.0:
        return probs
    
    # Get shape information
    batch_size, seq_len, vocab_size = probs.shape
    
    # Reshape the probabilities to handle properly
    flat_probs = probs.reshape(-1, vocab_size)  # Combine batch and seq dimensions
    
    # Sort probabilities in descending order
    sorted_probs = -jnp.sort(-flat_probs, axis=-1)  # Descending sort
    # Create indices tensor with the right shape
    indices = jnp.arange(vocab_size, dtype=jnp.int32)
    indices = jnp.broadcast_to(indices, flat_probs.shape)
    # Argsort to get indices in sorted order
    sorted_indices = jnp.argsort(-flat_probs, axis=-1)
    
    # Compute cumulative probabilities
    cumprobs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Create mask for probabilities within the nucleus
    nucleus_mask = cumprobs <= p
    
    # Always keep at least the top probability
    nucleus_mask = nucleus_mask.at[:, 0].set(True)
    
    # Apply the mask and normalize
    sorted_masked_probs = jnp.where(nucleus_mask, sorted_probs, 0.0)
    
    # Create tensor to hold the reordered probabilities
    reordered_probs = jnp.zeros_like(flat_probs)
    
    # This loop is needed to properly handle the scattering operation in a TPU-compatible way
    def apply_mask_for_batch(i, reordered):
        # For each item in the batch, place the sorted probabilities back in original order
        idx = sorted_indices[i]
        masked_probs = sorted_masked_probs[i]
        # Use scatter to place probabilities back in original order
        updated = reordered.at[i].scatter_update(idx, masked_probs)
        return updated
    
    # Apply masking for each item in batch
    reordered_probs = jax.lax.fori_loop(
        0, flat_probs.shape[0], apply_mask_for_batch, reordered_probs
    )
    
    # Normalize
    row_sums = reordered_probs.sum(axis=-1, keepdims=True)
    reordered_probs = reordered_probs / (row_sums + 1e-9)
    
    # Reshape back to original dimensions
    return reordered_probs.reshape(batch_size, seq_len, vocab_size) 