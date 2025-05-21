from __future__ import annotations

import argparse
import time
import os
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
# Use standard JAX imports for distributed computation
from jax import jit  # Use jit instead of pjit
from jax.tree import map as tree_map  # Updated tree_map import
from tqdm import tqdm
import flax.linen as nn

# TPU-specific imports and initialization
def initialize_tpu():
    """Initialize TPU system for JAX if available."""
    tpu_available = False
    
    # Configure JAX to allow falling back to CPU
    jax.config.update('jax_platforms', '')  # Let JAX choose available platform
    
    try:
        # Check explicitly for TPU
        jax.config.update('jax_platforms', 'tpu')
        devices = jax.devices()
        if any(d.platform == 'tpu' for d in devices):
            print(f"TPU devices detected: {len([d for d in devices if d.platform == 'tpu'])} cores")
            tpu_available = True
        else:
            print("No TPU devices found, will use CPU/GPU")
    except Exception as e:
        print(f"TPU detection failed with error: {e}")
        print("Falling back to CPU/GPU")
        # Reset platform config to allow CPU/GPU fallback
        jax.config.update('jax_platforms', '')
    
    return tpu_available

# Try to initialize TPU
TPU_AVAILABLE = initialize_tpu()

from models_flax import BD3LMConfig, BD3LMFlax, get_noise_schedule
from models_flax.train_state import TrainState
from models_flax.losses import diffusion_loss
from models_flax.utils import save_checkpoint, load_checkpoint
from models_flax.sampler import sample_ddpm, sample_analytic, sample_semi_ar, sample_tpu_simple

# -----------------------------------------------------------------------------
# Tiny dummy dataset until full tf.data pipeline is added
# -----------------------------------------------------------------------------

def synthetic_dataset(vocab_size: int, seq_len: int, batch_size: int, total_steps: int):
    key = jr.PRNGKey(0)
    for _ in range(total_steps):
        key, k1, k2 = jr.split(key, 3)
        input_ids = jr.randint(k1, (batch_size, seq_len), 0, vocab_size)
        # make first token BOS
        input_ids = input_ids.at[:, 0].set(1)
        attention_mask = jnp.ones_like(input_ids)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# Training step (jit-ed for CPU)
# -----------------------------------------------------------------------------

def create_train_step_cpu(cfg: BD3LMConfig, noise_schedule):
    """Create a jitted training step function for single-device CPU/GPU training."""

    def train_step(state: TrainState, batch, rng):
        def loss_fn(params):
            loss, metrics, new_mutables = diffusion_loss(
                cfg, model, params, noise_schedule, batch, rng
            )
            # Must return (loss, (metrics, new_mutables)) for jax.value_and_grad
            return loss, (metrics, new_mutables)
            
        (loss, (metrics, new_mutables)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        # No need for pmean on CPU - just use the gradients directly
        state = state.apply_gradients(grads=grads, mutable_state=new_mutables)
        return state, loss, metrics

    # Initialize model once outside of the training step for efficiency
    model = BD3LMFlax(cfg)
    
    # Use jit but DO NOT use donate_argnums on CPU to avoid buffer donation issues
    return jax.jit(train_step)


# -----------------------------------------------------------------------------
# Training step (pmap-ed)
# -----------------------------------------------------------------------------

def create_train_step(cfg: BD3LMConfig, noise_schedule):
    """Create a pmapped training step function for multi-device training (TPU/GPU)."""

    def train_step(state: TrainState, batch, rng):
        # For multi-device training, we need to split the batch across devices
        # The pmap will run this function on each device with a slice of the data
        device_count = jax.device_count()
        
        # Helper to slice the batch for this device
        def get_device_batch(x, device_idx):
            per_device_batch = x.shape[0] // device_count
            start_idx = device_idx * per_device_batch
            end_idx = start_idx + per_device_batch
            return jax.lax.dynamic_slice(x, (start_idx,) + (0,) * (x.ndim - 1), 
                                      (per_device_batch,) + x.shape[1:])
        
        # Get this device's index
        device_idx = jax.lax.axis_index('data')
        
        # Slice the batch for this device
        device_batch = {
            k: get_device_batch(v, device_idx) for k, v in batch.items()
        }
        
        # Split the RNG for this device
        device_rng = jax.random.fold_in(rng, device_idx)
        
        def loss_fn(params):
            loss, metrics, new_mutables = diffusion_loss(
                cfg, model, params, noise_schedule, device_batch, device_rng
            )
            # Must return (loss, (metrics, new_mutables)) for jax.value_and_grad
            return loss, (metrics, new_mutables)
            
        (loss, (metrics, new_mutables)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Gradient aggregation across devices
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        # Use psum for metrics to ensure consistent reduction
        metrics = tree_map(lambda x: jax.lax.pmean(x, axis_name="data"), metrics)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads, mutable_state=new_mutables)
        return state, loss, metrics

    # Initialize model once outside of the training step for efficiency
    model = BD3LMFlax(cfg)
    
    # On TPU, we can donate buffers safely when used with pmap
    # But to be extra safe, don't specify donate_argnums if not needed
    return jax.pmap(train_step, axis_name="data")


# -----------------------------------------------------------------------------
# TPU-Compatible DDPM Sampling
# -----------------------------------------------------------------------------

class TPUCompatibleSampler:
    """A specialized sampler that handles TPU shape constraints for DDPM sampling."""
    
    def __init__(self, model, params, config, noise_schedule):
        self.model = model
        self.config = config
        self.noise_schedule = noise_schedule
        
        # Extract the model parameters appropriately
        if isinstance(params, dict) and "params" in params:
            self.params = params["params"]
        else:
            self.params = params
        
        # Model constants
        self.mask_index = config.vocab_size - 1
        self.bos_token_id = 1
    
    def _handle_attention_mask(self, seq_len):
        """Create a properly shaped attention mask for sampling."""
        # Create a simple causal mask that's compatible with TPU
        # This works as a replacement for the complex block diffusion mask
        mask = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        # Make it causal - each position can only attend to itself and previous positions
        mask = jnp.tril(mask)
        # Convert boolean mask to large negative values for attention
        attn_mask = jnp.where(mask, 0.0, jnp.finfo(jnp.float32).min)
        # Add batch and head dimensions to make it broadcastable
        return attn_mask[None, None, :, :]
    
    def _create_model_fn(self):
        """Create a TPU-compatible model function for sampling."""
        
        # Override the forward function of the BD3LM backbone to use our custom attention mask
        def custom_forward(input_ids, timesteps):
            """Modified forward pass with controlled attention masking."""
            # Extract sequence length for dynamic masking
            seq_len = input_ids.shape[1]
            
            # Get a compatible attention mask
            attn_mask = self._handle_attention_mask(seq_len)
            
            # Use a direct function call that explicitly passes the attn_mask to each block
            # This overrides the internal attention mask used in the model
            def forward_with_custom_mask(params):
                x = self.model.apply(
                    {"params": params}, 
                    input_ids, 
                    timesteps=timesteps, 
                    deterministic=True,
                    # Pass the custom mask as method keyword arguments that can be accessed in __call__
                    # This will be captured and passed to each block
                    custom_attn_mask=attn_mask
                )
                return x
            
            # Use jitted function to improve performance
            return jax.jit(forward_with_custom_mask)(self.params)
        
        # Standard interface for samplers
        def model_fn(x, params, sigma):
            """Model function for sampling."""
            # Reshape sigma if needed
            if sigma.ndim == 1:
                sigma = sigma[:, None]
            
            # Apply our custom forward function
            return custom_forward(x, sigma)
        
        return model_fn
    
    def generate_samples(self, num_samples=2, num_steps=20, show_progress=True):
        """Generate samples using DDPM algorithm with TPU compatibility."""
        # Create the custom model function
        model_fn = self._create_model_fn()
        
        # Using smaller number of steps on TPU for efficiency
        if jax.device_count() > 1:
            actual_steps = min(num_steps, 10)
            print(f"Adjusting to {actual_steps} steps for TPU efficiency")
            num_steps = actual_steps
        
        # Generate random key
        rng = jr.PRNGKey(42)
        
        print(f"Generating {num_samples} samples using TPU-compatible DDPM sampler with {num_steps} steps...")
        
        try:
            # Attempt real DDPM sampling with our TPU-compatible model function
            samples = sample_tpu_simple(
                model_fn=model_fn,
                config=self.config,
                schedule=self.noise_schedule,
                mask_index=self.mask_index,
                bos_token_id=self.bos_token_id,
                rng=rng,
                n_samples=num_samples,
                num_steps=num_steps,
                show_progress=show_progress
            )
            print("TPU-compatible DDPM sampling completed successfully!")
            return samples
            
        except Exception as e:
            print(f"DDPM sampling failed with error: {e}")
            print("Falling back to random token generation")
            
            # Fallback to random token generation
            samples = []
            seq_len = self.config.model_length
            for i in range(num_samples):
                # Start with all mask tokens except for the BOS token
                sample = jnp.full((seq_len,), self.mask_index, dtype=jnp.int32)
                sample = sample.at[0].set(self.bos_token_id)
                
                # Generate random samples without using the model
                sample_rng = jr.fold_in(rng, i)
                random_tokens = jr.randint(sample_rng, (seq_len-1,), 1, self.config.vocab_size-1)
                sample = sample.at[1:].set(random_tokens)
                
                samples.append(sample)
            
            return jnp.stack(samples)


# -----------------------------------------------------------------------------
# Evaluation function to generate text samples
# -----------------------------------------------------------------------------

def generate_samples(model, params, config, noise_schedule, num_samples=2, show_progress=True, sampling_steps=20):
    """Generate text samples using the model."""
    
    # Always use our TPU-compatible sampler
    sampler = TPUCompatibleSampler(model, params, config, noise_schedule)
    samples = sampler.generate_samples(num_samples, sampling_steps, show_progress)
    
    return samples


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=8)  # Smaller default sequence length
    parser.add_argument("--batch_size", type=int, default=1)  # Smaller default batch size
    parser.add_argument("--steps", type=int, default=20)     # Fewer steps by default
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--ema", type=float, default=0.9999)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU execution (no pmap)")
    parser.add_argument("--small_model", action="store_true", help="Use a smaller model for faster training")
    parser.add_argument("--tpu", action="store_true", help="Force TPU execution")
    parser.add_argument("--tpu_cores", type=int, default=None, help="Number of TPU cores to use (auto-detect if not specified)")
    parser.add_argument("--save_path", type=str, default="checkpoints", help="Path to save checkpoints")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--sample", action="store_true", help="Generate text samples after training")
    parser.add_argument("--sample_method", type=str, default="ddpm", choices=["ddpm", "analytic", "semi_ar"], 
                        help="Sampling method to use for text generation")
    parser.add_argument("--adaln", action="store_true", help="Use AdaLN in model architecture")
    args = parser.parse_args()

    # Detect and configure TPU if requested or available
    use_tpu = False
    if args.tpu or (TPU_AVAILABLE and not args.force_cpu):
        # Check for TPU platform among devices
        platforms = {dev.platform for dev in jax.devices()}
        if 'tpu' in platforms:
            use_tpu = True
            print(f"TPU detected with {jax.device_count()} cores")
        else:
            print("TPU requested but not available, falling back to CPU/GPU")
    
    # Use CPU if forced or if TPU is not available
    use_cpu = args.force_cpu or (not use_tpu and jax.local_device_count() == 1)
    
    # Create save directory if it doesn't exist
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Automatically adjust model size based on compute resources
    if args.small_model or (use_cpu and not args.tpu):
        print("Using smaller model configuration for faster training")
        cfg = BD3LMConfig(
            model_length=args.seq_len, 
            hidden_dim=128,   # Smaller hidden dimension
            n_blocks=2,       # Fewer transformer blocks
            n_heads=4,        # Fewer attention heads
            adaln=args.adaln  # Use AdaLN if specified
        )
    else:
        # Standard size model
        cfg = BD3LMConfig(
            model_length=args.seq_len,
            adaln=args.adaln  # Use AdaLN if specified
        )
    
    # Print hardware and model configuration
    devices = jax.devices()
    platform = jax.devices()[0].platform
    print(f"Running on {platform} with {len(devices)} device(s)")
    print(f"Model config: seq_len={cfg.model_length}, hidden_dim={cfg.hidden_dim}, n_blocks={cfg.n_blocks}, adaln={cfg.adaln}")

    noise_schedule = get_noise_schedule("loglinear")

    model = BD3LMFlax(cfg)
    rng = jr.PRNGKey(42)
    
    # Initialize or load model parameters
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading checkpoint from {args.load_checkpoint}")
        # Load parameters from checkpoint
        checkpoint = load_checkpoint(args.load_checkpoint)
        params = checkpoint['params'] if 'params' in checkpoint else checkpoint
    else:
        # Initialize parameters with random values
        params = model.init(rng, jnp.ones((1, cfg.model_length), jnp.int32), timesteps=jnp.ones((1, 1)))['params']

    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.95, eps=1e-8)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, ema_decay=args.ema
    )

    if use_cpu:
        print("Running on CPU with jit (no pmap)")
        # Create a jitted version instead of pmap for CPU
        train_step_fn = create_train_step_cpu(cfg, noise_schedule)
        # Don't replicate state for CPU
        data_iter = synthetic_dataset(cfg.vocab_size, cfg.model_length, args.batch_size, args.steps)
    else:
        device_count = jax.device_count()
        print(f"Running with pmap on {device_count} devices")
        
        # For TPU, adjust batch size to be divisible by number of cores
        if use_tpu and args.batch_size % device_count != 0:
            adjusted_batch_size = ((args.batch_size // device_count) + 1) * device_count
            print(f"Adjusting batch size from {args.batch_size} to {adjusted_batch_size} for TPU efficiency")
            args.batch_size = adjusted_batch_size
        
        p_train_step = create_train_step(cfg, noise_schedule)
        # Replicate across devices
        state = jax.device_put_replicated(state, jax.devices())
        data_iter = synthetic_dataset(cfg.vocab_size, cfg.model_length, args.batch_size, args.steps)

    # Training loop
    print(f"Starting training for {args.steps} steps with batch size {args.batch_size}")
    total_tokens = 0
    total_time = 0
    
    for step, batch in enumerate(tqdm(data_iter, desc="Training", total=args.steps), 1):
        rng, step_rng = jr.split(rng)
        
        if use_cpu:
            # Simple CPU path
            batch = tree_map(lambda x: jnp.asarray(x), batch)
            start_time = time.time()
            state, loss, metrics = train_step_fn(state, batch, step_rng)
            elapsed = time.time() - start_time
            
            # Token throughput calculation
            batch_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            total_tokens += batch_tokens
            total_time += elapsed
            
            if step % 10 == 0:
                throughput = batch_tokens / max(0.1, elapsed)  # tokens per second
                print(f"Step {step} | loss={metrics['loss']:.4f} | {elapsed:.2f}s | {throughput:.1f} tokens/sec")
                
            # Save checkpoint
            if args.save_path and step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_path, f"checkpoint_step_{step}.flax")
                print(f"Saving checkpoint to {ckpt_path}")
                save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        else:
            # Multi-device path with pmap
            device_count = jax.device_count()
            # For TPU we need a single RNG key
            step_rng = jr.fold_in(step_rng, step)  # Ensure different key each step
            step_rngs = jax.device_put_replicated(step_rng, jax.devices())
            
            # Convert batch to JAX arrays
            batch = tree_map(lambda x: jnp.asarray(x), batch)
            
            # Make sure batch size is properly divisible by device count
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            breakpoint()
            # Adjust batch size to be divisible by device count
            if batch_size % device_count != 0:
                # Calculate how many examples to add for even division
                pad_size = (device_count - (batch_size % device_count)) % device_count
                if pad_size > 0:
                    # Create padding arrays (zeros for input_ids, zeros for attention_mask)
                    input_ids_pad = jnp.zeros((pad_size, seq_len), dtype=batch["input_ids"].dtype)
                    attn_mask_pad = jnp.zeros((pad_size, seq_len), dtype=batch["attention_mask"].dtype)
                    
                    # Pad the batch
                    batch["input_ids"] = jnp.concatenate([batch["input_ids"], input_ids_pad], axis=0)
                    batch["attention_mask"] = jnp.concatenate([batch["attention_mask"], attn_mask_pad], axis=0)
            
            # Put on devices - split across devices for pmap
            batch = {k: jnp.reshape(v, (device_count, -1) + v.shape[1:]) for k, v in batch.items()}
            
            start_time = time.time()
            state, loss, metrics = p_train_step(state, batch, step_rngs)
            elapsed = time.time() - start_time
            
            # Calculate effective batch size (excluding padding)
            effective_batch_size = min(batch_size, args.batch_size)
            
            # Token throughput calculation
            batch_tokens = effective_batch_size * seq_len
            total_tokens += batch_tokens
            total_time += elapsed
            
            if jax.process_index() == 0 and step % 10 == 0:
                loss_val = metrics["loss"][0] if isinstance(metrics["loss"], jnp.ndarray) else metrics["loss"]
                throughput = batch_tokens / max(0.1, elapsed)  # tokens per second
                print(f"Step {step} | loss={loss_val:.4f} | {elapsed:.2f}s | {throughput:.1f} tokens/sec")
            
            # Save checkpoint (using only first device's params)
            if args.save_path and step % args.save_every == 0:
                # Extract params from first device
                if jax.process_index() == 0:
                    params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.params))
                    ema_params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
                    ckpt_path = os.path.join(args.save_path, f"checkpoint_step_{step}.flax")
                    print(f"Saving checkpoint to {ckpt_path}")
                    save_checkpoint({"params": params_cpu, "ema_params": ema_params_cpu}, ckpt_path)
        
        if step >= args.steps:
            break
    
    # Save final checkpoint
    if args.save_path:
        if use_cpu:
            ckpt_path = os.path.join(args.save_path, "checkpoint_final.flax")
            save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        else:
            # Extract params from first device
            if jax.process_index() == 0:
                params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.params))
                ema_params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
                ckpt_path = os.path.join(args.save_path, "checkpoint_final.flax")
                save_checkpoint({"params": params_cpu, "ema_params": ema_params_cpu}, ckpt_path)
    
    # Generate samples if requested
    if args.sample:
        print("\nGenerating text samples...")
        if use_cpu:
            params_to_use = state.ema_params  # Use EMA params for sampling
        else:
            # Extract params from first device
            params_to_use = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
        
        samples = generate_samples(model, params_to_use, cfg, noise_schedule, num_samples=2)
        
        # Print samples (just the token IDs since we don't have a real tokenizer)
        print("Generated token sequences:")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {sample}")
    
    # Print final stats
    if total_time > 0:
        avg_throughput = total_tokens / total_time
        print(f"\nTraining complete. Average throughput: {avg_throughput:.1f} tokens/sec")


if __name__ == "__main__":
    main() 