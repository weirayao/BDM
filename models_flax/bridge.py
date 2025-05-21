from __future__ import annotations

"""Bridge module for integrating JAX/Flax implementation with PyTorch.

This module provides utility functions and classes to easily convert and use
Flax models with the existing PyTorch infrastructure in main.py and diffusion.py.
"""

import os
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import transformers
from functools import partial

from .bd3lm_arch import BD3LMConfig, BD3LMFlax
from .noise_schedule import get_noise as get_noise_schedule, NoiseSchedule
from .utils import (
    convert_pytorch_state_dict_to_flax,
    convert_flax_state_dict_to_pytorch,
    save_checkpoint,
    load_checkpoint
)
from .sampler import sample_ddpm, sample_analytic, sample_semi_ar


class FlaxModelWrapper(nn.Module):
    """Wrapper class for using Flax models with PyTorch infrastructure.
    
    This class wraps a Flax model to make it compatible with the PyTorch
    interface used in main.py and diffusion.py. It handles the conversion
    between PyTorch tensors and JAX arrays, and provides methods for inference
    and sampling.
    """

    def __init__(
        self,
        config: Union[Dict, BD3LMConfig],
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the wrapper with a Flax model.
        
        Args:
            config: Model configuration
            model_path: Path to the model checkpoint
            device: Device to use for PyTorch tensors
            dtype: Data type to use for PyTorch tensors
        """
        super().__init__()
        
        # Convert dictionary config to BD3LMConfig if needed
        if isinstance(config, dict):
            self.config = BD3LMConfig(**config)
        else:
            self.config = config
            
        # Convert PyTorch dtype to JAX dtype
        if dtype == torch.float32:
            self.jax_dtype = jnp.float32
        elif dtype == torch.float16:
            self.jax_dtype = jnp.float16
        elif dtype == torch.bfloat16:
            self.jax_dtype = jnp.bfloat16
        else:
            self.jax_dtype = jnp.float32
            
        # Override config dtype with the one provided
        self.config = self.config.replace(dtype=self.jax_dtype)
        
        # Initialize the Flax model
        self.model = BD3LMFlax(self.config)
        
        # Initialize model parameters
        self.params = None
        if model_path and os.path.exists(model_path):
            self.params = load_checkpoint(model_path)
        else:
            # Initialize parameters with random values
            dummy_input = jnp.ones((1, self.config.model_length), dtype=jnp.int32)
            dummy_timesteps = jnp.ones((1, 1), dtype=jnp.float32)
            self.params = self.model.init(jax.random.PRNGKey(0), dummy_input, timesteps=dummy_timesteps)
            
        # Initialize noise schedule
        self.noise_schedule = get_noise_schedule("loglinear")
        
        # Set device for PyTorch operations
        self.device = device
        self.dtype = dtype
        
        # Create model inference function
        def model_fn(x, params, sigma):
            """Model inference function for sampling."""
            # Reshape sigma if needed
            if sigma.ndim == 1:
                sigma = sigma[:, None]
                
            # Call the forward method of the model
            return self.model.apply(
                {"params": params["params"]},
                x,
                timesteps=sigma,
                deterministic=True
            )
            
        self.model_fn = model_fn
        
    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        sample_mode: bool = False,
        store_kv: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            sigma: Noise level
            sample_mode: Whether to use sampling mode
            store_kv: Whether to store KV cache
            
        Returns:
            logits: Model logits
        """
        # Convert PyTorch tensors to JAX arrays
        input_ids_jax = jnp.array(input_ids.cpu().numpy())
        
        # Default sigma if not provided
        if sigma is None:
            sigma = torch.zeros((input_ids.shape[0], 1), device=self.device)
            sigma_jax = jnp.zeros((input_ids.shape[0], 1))
        else:
            sigma_jax = jnp.array(sigma.cpu().numpy())
            
        # Call the model
        logits = self.model.apply(
            {"params": self.params["params"]},
            input_ids_jax,
            timesteps=sigma_jax,
            deterministic=True
        )
        
        # Convert logits back to PyTorch tensor
        logits_torch = torch.tensor(np.array(logits), device=self.device, dtype=self.dtype)
        return logits_torch
    
    def save_pretrained(self, save_dir: str) -> None:
        """Save the model to a directory.
        
        Args:
            save_dir: Directory to save the model
        """
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(self.params, os.path.join(save_dir, "model.flax"))
        
        # Also save the config
        config_dict = {
            "model_length": self.config.model_length,
            "hidden_dim": self.config.hidden_dim,
            "n_blocks": self.config.n_blocks,
            "n_heads": self.config.n_heads,
            "block_size": self.config.block_size,
            "vocab_size": self.config.vocab_size,
            "cross_attn": self.config.cross_attn,
            "adaln": self.config.adaln,
            "causal": self.config.causal,
            "attn_backend": self.config.attn_backend,
            "dropout": self.config.dropout,
            "time_conditioning": self.config.time_conditioning,
            "var_min": self.config.var_min,
            "sampling_eps_min": self.config.sampling_eps_min,
            "sampling_eps_max": self.config.sampling_eps_max,
        }
        
        import json
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "FlaxModelWrapper":
        """Load the model from a pretrained checkpoint.
        
        Args:
            pretrained_model_path: Path to the pretrained model
            device: Device to use for PyTorch tensors
            dtype: Data type to use for PyTorch tensors
            
        Returns:
            model: Loaded model
        """
        # Load config
        import json
        with open(os.path.join(pretrained_model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
            
        # Create config object
        config = BD3LMConfig(**config_dict)
        
        # Load model
        model_path = os.path.join(pretrained_model_path, "model.flax")
        return cls(config, model_path, device, dtype)
    
    def from_pytorch_model(self, pytorch_model: nn.Module) -> None:
        """Load parameters from a PyTorch model.
        
        Args:
            pytorch_model: PyTorch model to load parameters from
        """
        # Get PyTorch state dict
        pt_state_dict = pytorch_model.state_dict()
        
        # Convert to Flax parameters
        self.params = convert_pytorch_state_dict_to_flax(pt_state_dict, self.model, self.config)
    
    def to_pytorch_model(self, pytorch_model: nn.Module) -> nn.Module:
        """Convert parameters to a PyTorch model.
        
        Args:
            pytorch_model: PyTorch model to load parameters into
            
        Returns:
            pytorch_model: Updated PyTorch model
        """
        # Convert Flax parameters to PyTorch state dict
        pt_state_dict = convert_flax_state_dict_to_pytorch(self.params, pytorch_model, self.config)
        
        # Load state dict into PyTorch model
        pytorch_model.load_state_dict(pt_state_dict)
        return pytorch_model
    
    def sample(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        n_samples: int = 1,
        num_steps: int = 50,
        sampler: str = "ddpm",
        nucleus_p: float = 0.9,
        first_hitting: bool = False,
        eps: float = 1e-5,
        show_progress: bool = True,
    ) -> List[str]:
        """Generate samples from the model.
        
        Args:
            tokenizer: Tokenizer to use for decoding
            n_samples: Number of samples to generate
            num_steps: Number of sampling steps
            sampler: Sampling method to use (ddpm, analytic, or semi_ar)
            nucleus_p: Nucleus sampling parameter
            first_hitting: Whether to use first hitting time sampling
            eps: Minimum time value
            show_progress: Whether to show a progress bar
            
        Returns:
            samples: Generated text samples
        """
        # Determine the mask index
        if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
            mask_index = tokenizer.mask_token_id
        else:
            mask_index = self.config.vocab_size - 1
            
        # Determine the BOS token ID
        bos_token_id = tokenizer.bos_token_id
        
        # Create random key
        rng = jax.random.PRNGKey(int(torch.randint(0, 2**30, (1,)).item()))
        
        # Choose the sampling method
        if sampler == "analytic":
            sample_fn = sample_analytic
            kwargs = {
                "model_fn": self.model_fn,
                "params": self.params,
                "config": self.config,
                "schedule": self.noise_schedule,
                "mask_index": mask_index,
                "bos_token_id": bos_token_id,
                "rng": rng,
                "n_samples": n_samples,
                "num_steps": num_steps,
                "eps": eps,
                "nucleus_p": nucleus_p,
                "show_progress": show_progress,
            }
        elif sampler == "semi_ar":
            sample_fn = sample_semi_ar
            kwargs = {
                "model_fn": self.model_fn,
                "params": self.params,
                "config": self.config,
                "schedule": self.noise_schedule,
                "mask_index": mask_index,
                "bos_token_id": bos_token_id,
                "rng": rng,
                "n_samples": n_samples,
                "num_steps": num_steps,
                "eps": eps,
                "block_size": self.config.block_size,
                "nucleus_p": nucleus_p,
                "first_hitting": first_hitting,
                "show_progress": show_progress,
            }
        else:  # Default to DDPM
            sample_fn = sample_ddpm
            kwargs = {
                "model_fn": self.model_fn,
                "params": self.params,
                "config": self.config,
                "schedule": self.noise_schedule,
                "mask_index": mask_index,
                "bos_token_id": bos_token_id,
                "rng": rng,
                "n_samples": n_samples,
                "num_steps": num_steps,
                "eps": eps,
                "block_size": self.config.block_size,
                "nucleus_p": nucleus_p,
                "first_hitting": first_hitting,
                "show_progress": show_progress,
            }
            
        # Generate samples
        samples = sample_fn(**kwargs)
        
        # Decode samples
        decoded_samples = tokenizer.batch_decode(np.array(samples), skip_special_tokens=True)
        return decoded_samples


class HuggingFaceModelWrapper(FlaxModelWrapper):
    """Wrapper for exporting the model to HuggingFace format.
    
    This class provides additional functionality for loading and saving
    models in HuggingFace format.
    """
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        **kwargs
    ) -> "HuggingFaceModelWrapper":
        """Load from HuggingFace Hub or local directory."""
        # If it's a local path and has our specific format
        if os.path.exists(os.path.join(pretrained_model_name_or_path, "model.flax")):
            return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Otherwise, assume it's a HuggingFace model
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Convert HuggingFace config to our config
        our_config = {
            "model_length": getattr(config, "max_position_embeddings", 1024),
            "hidden_dim": config.hidden_size,
            "n_blocks": config.num_hidden_layers,
            "n_heads": config.num_attention_heads,
            "block_size": getattr(config, "block_size", 1),
            "vocab_size": config.vocab_size,
            "cross_attn": getattr(config, "cross_attn", True),
            "adaln": getattr(config, "adaln", True),
            "causal": getattr(config, "causal", False),
            "attn_backend": getattr(config, "attn_backend", "sdpa"),
            "dropout": getattr(config, "hidden_dropout_prob", 0.1),
            "time_conditioning": getattr(config, "time_conditioning", False),
            "var_min": getattr(config, "var_min", True),
            "sampling_eps_min": getattr(config, "sampling_eps_min", 1e-3),
            "sampling_eps_max": getattr(config, "sampling_eps_max", 0.999),
        }
        
        # Create the model
        model = cls(our_config, *args, **kwargs)
        
        # Load model weights from HuggingFace
        hf_model = transformers.AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        
        # Convert parameters
        model.from_pytorch_model(hf_model)
        
        return model
    
    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = False,
        token: Optional[str] = None,
    ) -> None:
        """Push the model to the HuggingFace Hub.
        
        Args:
            repo_id: Repository ID on the Hub
            commit_message: Commit message
            private: Whether the repository should be private
            token: HuggingFace token
        """
        # Create a temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save the model
            self.save_pretrained(tmp_dir)
            
            # Convert to HuggingFace format
            config = transformers.PretrainedConfig(
                hidden_size=self.config.hidden_dim,
                num_hidden_layers=self.config.n_blocks,
                num_attention_heads=self.config.n_heads,
                max_position_embeddings=self.config.model_length,
                vocab_size=self.config.vocab_size,
                hidden_dropout_prob=self.config.dropout,
                attention_probs_dropout_prob=self.config.dropout,
                model_type="bd3lm",
                block_size=self.config.block_size,
                cross_attn=self.config.cross_attn,
                adaln=self.config.adaln,
                causal=self.config.causal,
                attn_backend=self.config.attn_backend,
                time_conditioning=self.config.time_conditioning,
                var_min=self.config.var_min,
                sampling_eps_min=self.config.sampling_eps_min,
                sampling_eps_max=self.config.sampling_eps_max,
            )
            
            # Save HuggingFace config
            config.save_pretrained(tmp_dir)
            
            # Create a dummy PyTorch model with the right structure
            from transformers.models.bert.modeling_bert import BertForMaskedLM
            dummy_model = BertForMaskedLM(config)
            
            # Convert our parameters to PyTorch
            pytorch_model = self.to_pytorch_model(dummy_model)
            
            # Save PyTorch model
            pytorch_model.save_pretrained(tmp_dir)
            
            # Push to Hub
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id, private=private, exist_ok=True)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                commit_message=commit_message,
            ) 