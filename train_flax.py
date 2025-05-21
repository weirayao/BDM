from __future__ import annotations

import argparse
import time
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import functools

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
# Use standard JAX imports for distributed computation
from jax import jit, pmap
try:
    from jax import pjit  # Try to import from main namespace (newer JAX)
except ImportError:
    try:
        from jax.experimental import pjit  # Try experimental module (mid-version JAX)
    except ImportError:
        from jax.experimental.pjit import pjit  # Fall back to older version
from jax.sharding import PositionalSharding, NamedSharding
from jax.experimental import mesh_utils
from jax.tree import map as tree_map  # Updated tree_map import
from tqdm import tqdm
import flax.linen as nn
import numpy as np
import tensorflow as tf

# TPU-specific imports and initialization
def initialize_distributed():
    """Initialize JAX for distributed training on Cloud TPUs.
    
    Returns:
        Tuple of (process_count, process_index, local_device_count, global_device_count)
    """
    # Initialize JAX distributed
    try:
        jax.distributed.initialize()
        print("JAX distributed initialized successfully")
    except Exception as e:
        print(f"JAX distributed initialization failed: {e}")
        print("Falling back to single-process mode")
    
    # Get process and device information
    process_count = jax.process_count()
    process_index = jax.process_index()
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    
    if process_count > 1:
        print(f"Running in distributed mode with {process_count} processes")
        print(f"This is process {process_index} with {local_device_count} local devices")
        print(f"Total devices across all processes: {global_device_count}")
    else:
        print(f"Running in single-process mode with {local_device_count} devices")
    
    return process_count, process_index, local_device_count, global_device_count

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
# Initialize distributed training
PROCESS_COUNT, PROCESS_INDEX, LOCAL_DEVICE_COUNT, GLOBAL_DEVICE_COUNT = initialize_distributed()

from models_flax import BD3LMConfig, BD3LMFlax, get_noise_schedule
from models_flax.train_state import TrainState
from models_flax.losses import diffusion_loss
from models_flax.utils import save_checkpoint, load_checkpoint
from models_flax.sampler import sample_ddpm, sample_analytic, sample_semi_ar, sample_tpu_simple

# -----------------------------------------------------------------------------
# TensorFlow dataset pipeline for distributed training
# -----------------------------------------------------------------------------

def create_tf_dataset(vocab_size: int, seq_len: int, batch_size: int, total_steps: int, 
                      process_count: int = 1, process_index: int = 0, seed: int = 42):
    """Create a TensorFlow dataset that's sharded across processes for distributed training.
    
    Args:
        vocab_size: Size of the vocabulary
        seq_len: Sequence length for input data
        batch_size: Global batch size (will be divided among processes)
        total_steps: Total number of training steps
        process_count: Number of processes in distributed training
        process_index: Index of the current process
        seed: Random seed for reproducibility
    
    Returns:
        A TensorFlow dataset that yields batches of input data
    """
    # Set random seed for reproducibility
    tf.random.set_seed(seed + process_index)
    
    # Calculate per-process batch size
    per_process_batch_size = batch_size // process_count
    # Ensure at least 1 example per process
    per_process_batch_size = max(1, per_process_batch_size)
    
    # Function to generate random token IDs
    def generate_random_batch(_):
        # Generate random token IDs
        input_ids = tf.random.uniform(
            shape=(per_process_batch_size, seq_len),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32
        )
        # Set first token to BOS (token ID 1)
        input_ids = tf.concat([
            tf.ones((per_process_batch_size, 1), dtype=tf.int32),
            input_ids[:, 1:],
        ], axis=1)
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = tf.ones_like(input_ids)
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    # Create dataset from range
    dataset = tf.data.Dataset.range(total_steps * process_count)
    
    # Shard the dataset across processes
    dataset = dataset.shard(process_count, process_index)
    
    # Map to generate random batches
    dataset = dataset.map(
        generate_random_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def tf_dataset_to_jax_iterators(dataset, global_batch_size, steps_per_epoch):
    """Convert TensorFlow dataset to JAX iterators for multi-host training."""
    # Create a pipeline to convert TF tensors to NumPy arrays
    def to_numpy_pipeline(data):
        return {k: v.numpy() for k, v in data.items()}
    
    # Get Python iterator
    py_iterator = map(to_numpy_pipeline, dataset.as_numpy_iterator())
    
    # Function to convert data to JAX arrays and shard across devices
    def prepare_tf_data(structures):
        arrays = {k: jnp.asarray(v) for k, v in structures.items()}
        return arrays
    
    # Create iterator for JAX
    return (prepare_tf_data(batch) for batch in py_iterator)


# -----------------------------------------------------------------------------
# Synthetic dataset (kept for compatibility and single-process training)
# -----------------------------------------------------------------------------

def synthetic_dataset(vocab_size: int, seq_len: int, batch_size: int, total_steps: int):
    key = jr.PRNGKey(0)
    for _ in range(total_steps):
        key, k1, k2 = jr.split(key, 3)
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32) + 1
        # make first token BOS
        input_ids = input_ids.at[:, 0].set(1)
        attention_mask = jnp.ones_like(input_ids)
        yield {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# Training step with pjit for multi-host training
# -----------------------------------------------------------------------------

def setup_mesh(mesh_shape, mesh_axes):
    """Create a device mesh for distributed training."""
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, mesh_axes)


def create_mesh_pjit_train_step(cfg: BD3LMConfig, noise_schedule):
    """Create a pjit-based training step for multi-host training."""
    
    def train_step(state: TrainState, batch, rng):
        """Training step function for multi-host training."""
        def loss_fn(params):
            loss, metrics, new_mutables = diffusion_loss(
                cfg, model, params, noise_schedule, batch, rng
            )
            # Must return (loss, (metrics, new_mutables)) for jax.value_and_grad
            return loss, (metrics, new_mutables)
        
        (loss, (metrics, new_mutables)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # All-reduce gradients across all devices in all hosts
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch") 
        metrics = tree_map(lambda x: jax.lax.pmean(x, axis_name="batch"), metrics)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads, mutable_state=new_mutables)
        return state, loss, metrics
    
    # Initialize model
    model = BD3LMFlax(cfg)
    
    # Create a PartitionSpec to specify sharding constraints
    from jax.sharding import PartitionSpec as P
    
    # Define input and output specs for pjit
    state_spec = jax.tree.map(lambda _: P("batch"), TrainState.empty_node())
    batch_spec = jax.tree.map(lambda _: P("batch"), {"input_ids": None, "attention_mask": None})
    rng_spec = P(None)  # RNG keys aren't sharded
    
    # Create pjit-ed train step
    pjit_train_step = pjit(
        train_step,
        in_shardings=(state_spec, batch_spec, rng_spec),
        out_shardings=(state_spec, None, None),
        donate_argnums=(0,)  # Donate the state buffer for improved performance
    )
    
    return pjit_train_step


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
# Training step (pmap-ed for single-host multi-device)
# -----------------------------------------------------------------------------

def create_train_step(cfg: BD3LMConfig, noise_schedule):
    """Create a pmapped training step function for single-host multi-device training."""

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
        # Use pmean for metrics to ensure consistent reduction
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
# Evaluation function to generate text samples
# -----------------------------------------------------------------------------

def generate_samples(model, params, config, noise_schedule, num_samples=2, show_progress=True, sampling_steps=20, is_multi_host=False):
    """Generate text samples using the model.
    
    Args:
        model: The model to use for sampling
        params: Model parameters
        config: Model configuration
        noise_schedule: Noise schedule for diffusion sampling
        num_samples: Number of samples to generate
        show_progress: Whether to display a progress bar
        sampling_steps: Number of steps for DDPM sampling
        is_multi_host: Whether we're in a multi-host setup
        
    Returns:
        Generated token sequences
    """
    # Ensure we're on the main process if multi-host (sampling only happens on one process)
    if is_multi_host and jax.process_index() != 0:
        return None
    
    # Create a special function that wraps the model for sampling purposes
    # Use a simplified approach that's compatible with TPU constraints
    
    try:
        # Extract the model parameters if needed
        if isinstance(params, dict) and params.get("params") is not None:
            model_params = params["params"]
        else:
            model_params = params
        
        # Define a simplified model function that avoids shape issues
        def model_fn(x, params, sigma):
            """Model function that handles reshaping for compatibility."""
            # Reshape sigma if needed
            if sigma.ndim == 1:
                sigma = sigma[:, None]
            
            # Use direct function call to avoid shape issues
            return model.apply({"params": model_params}, x, timesteps=sigma, deterministic=True)
        
        # Set mask and BOS token indices
        mask_index = config.vocab_size - 1
        bos_token_id = 1
        
        # Generate random key
        rng = jr.PRNGKey(42)
        
        print(f"Generating {num_samples} samples using TPU-compatible sampler with {sampling_steps} steps...")
        
        # For TPU, use our specialized sampling approach
        if jax.device_count() > 1:
            # Run with smaller number of steps for TPU
            actual_steps = min(sampling_steps, 10)
            print(f"Adjusting to {actual_steps} steps for TPU efficiency")
            
            # Use the simplified TPU-compatible sampler
            samples = sample_tpu_simple(
                model_fn=model_fn,
                config=config,
                schedule=noise_schedule,
                mask_index=mask_index,
                bos_token_id=bos_token_id,
                rng=rng,
                n_samples=num_samples,
                num_steps=actual_steps,
                show_progress=show_progress
            )
            print("TPU-compatible sampling completed successfully!")
        else:
            # For CPU/GPU, use standard DDPM
            samples = sample_ddpm(
                model_fn=model_fn,
                params=None,  # Not used directly
                config=config,
                schedule=noise_schedule,
                mask_index=mask_index,
                bos_token_id=bos_token_id,
                rng=rng,
                n_samples=num_samples,
                num_steps=sampling_steps,
                show_progress=show_progress
            )
            print("Sampling completed successfully!")
            
        return samples
    except Exception as e:
        print(f"Error during sampling: {e}")
        print("Falling back to simple test sequence generation")
        
        # Generate simple test sequences as fallback
        samples = []
        for i in range(num_samples):
            sample = jnp.arange(1, config.model_length + 1) % (config.vocab_size - 2) + 1
            samples.append(sample)
        
        return jnp.stack(samples)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=8)  # Smaller default sequence length
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller default batch size
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
    parser.add_argument("--multi_host", action="store_true", help="Enable multi-host training (for Cloud TPU)")
    parser.add_argument("--process_count", type=int, default=None, 
                        help="Override detected process count for testing")
    parser.add_argument("--log_every", type=int, default=10, help="Log metrics every N steps")
    args = parser.parse_args()

    # Detect hardware configuration
    if args.process_count is not None:
        process_count = args.process_count
        process_index = PROCESS_INDEX
    else:
        process_count = PROCESS_COUNT 
        process_index = PROCESS_INDEX
    
    local_device_count = LOCAL_DEVICE_COUNT
    global_device_count = GLOBAL_DEVICE_COUNT
    
    # Only the first process should print certain messages
    is_main_process = process_index == 0
    
    # Multi-host training enabled for TPU or explicitly requested
    use_multi_host = args.multi_host or (TPU_AVAILABLE and process_count > 1)
    
    # Log detailed hardware configuration on main process
    if is_main_process:
        print(f"Hardware configuration:")
        print(f"  Process count: {process_count}")
        print(f"  Local devices per process: {local_device_count}")
        print(f"  Total devices: {global_device_count}")
        print(f"  Multi-host training: {'enabled' if use_multi_host else 'disabled'}")

    # Detect and configure TPU if requested or available
    use_tpu = False
    if args.tpu or (TPU_AVAILABLE and not args.force_cpu):
        # Check for TPU platform among devices
        platforms = {dev.platform for dev in jax.devices()}
        if 'tpu' in platforms:
            use_tpu = True
            if is_main_process:
                print(f"TPU detected with {global_device_count} total cores across {process_count} processes")
        elif is_main_process:
            print("TPU requested but not available, falling back to CPU/GPU")
    
    # Use CPU if forced or if TPU is not available
    use_cpu = args.force_cpu or (not use_tpu and local_device_count == 1)
    
    # Create save directory if it doesn't exist (only on main process)
    if args.save_path and is_main_process:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Automatically adjust model size based on compute resources
    if args.small_model or (use_cpu and not args.tpu):
        if is_main_process:
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
    
    # Print hardware and model configuration (main process only)
    if is_main_process:
        devices = jax.devices()
        platform = jax.devices()[0].platform
        print(f"Running on {platform} with {len(devices)} device(s) in this process")
        print(f"Model config: seq_len={cfg.model_length}, hidden_dim={cfg.hidden_dim}, n_blocks={cfg.n_blocks}, adaln={cfg.adaln}")

    noise_schedule = get_noise_schedule("loglinear")
    model = BD3LMFlax(cfg)
    
    # Create consistent RNG key across hosts
    global_seed = 42
    rng = jr.PRNGKey(global_seed)
    
    # Initialize or load model parameters
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if is_main_process:
            print(f"Loading checkpoint from {args.load_checkpoint}")
        # Load parameters from checkpoint
        checkpoint = load_checkpoint(args.load_checkpoint)
        params = checkpoint['params'] if 'params' in checkpoint else checkpoint
    else:
        # Initialize parameters with random values
        rng, init_rng = jr.split(rng)
        # Use same random seed across hosts for consistent initialization
        init_rng = jr.PRNGKey(global_seed + 1)
        params = model.init(init_rng, jnp.ones((1, cfg.model_length), jnp.int32), timesteps=jnp.ones((1, 1)))['params']

    # Create optimizer
    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.95, eps=1e-8)
    
    # Create train state
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, ema_decay=args.ema
    )

    # Setup for different training modes
    if use_cpu:
        if is_main_process:
            print("Running on CPU with jit (no pmap)")
        # Create a jitted version instead of pmap for CPU
        train_step_fn = create_train_step_cpu(cfg, noise_schedule)
        # Don't replicate state for CPU
        data_iter = synthetic_dataset(cfg.vocab_size, cfg.model_length, args.batch_size, args.steps)
    elif use_multi_host:
        if is_main_process:
            print(f"Running with multi-host training across {process_count} processes")
            print(f"Total TPU devices: {global_device_count}")
        
        # For multi-host TPU, set up a device mesh
        mesh_shape = (process_count, local_device_count)
        mesh_axes = ('batch', 'model')
        
        # Create mesh for distributed training
        device_mesh = setup_mesh(mesh_shape, ('process', 'device'))
        
        # Create pjit-based train step
        pjit_train_step = create_mesh_pjit_train_step(cfg, noise_schedule)
        
        # Adjust global batch size to be divisible by number of global devices
        global_batch_size = args.batch_size
        if global_batch_size % global_device_count != 0:
            adjusted_batch_size = ((global_batch_size // global_device_count) + 1) * global_device_count
            if is_main_process:
                print(f"Adjusting batch size from {global_batch_size} to {adjusted_batch_size} for TPU efficiency")
            global_batch_size = adjusted_batch_size
            
        # Create TensorFlow dataset sharded by process
        tf_dataset = create_tf_dataset(
            vocab_size=cfg.vocab_size,
            seq_len=cfg.model_length,
            batch_size=global_batch_size,
            total_steps=args.steps,
            process_count=process_count,
            process_index=process_index,
            seed=global_seed
        )
        
        # Convert to JAX iterators
        data_iter = tf_dataset_to_jax_iterators(tf_dataset, global_batch_size, args.steps)
    else:
        # Single-host, multi-device setup (standard pmap)
        device_count = jax.device_count()
        if is_main_process:
            print(f"Running with pmap on {device_count} devices (single host)")
        
        # For TPU, adjust batch size to be divisible by number of cores
        if use_tpu and args.batch_size % device_count != 0:
            adjusted_batch_size = ((args.batch_size // device_count) + 1) * device_count
            if is_main_process:
                print(f"Adjusting batch size from {args.batch_size} to {adjusted_batch_size} for TPU efficiency")
            args.batch_size = adjusted_batch_size
        
        p_train_step = create_train_step(cfg, noise_schedule)
        # Replicate across devices
        state = jax.device_put_replicated(state, jax.devices())
        data_iter = synthetic_dataset(cfg.vocab_size, cfg.model_length, args.batch_size, args.steps)

    # Training loop
    if is_main_process:
        print(f"Starting training for {args.steps} steps with batch size {args.batch_size}")
    total_tokens = 0
    total_time = 0
    
    # Use tqdm progress bar on main process only
    if is_main_process:
        progress_bar = tqdm(total=args.steps, desc="Training")
    
    for step, batch in enumerate(data_iter, 1):
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
            
            if is_main_process and step % args.log_every == 0:
                throughput = batch_tokens / max(0.1, elapsed)  # tokens per second
                print(f"Step {step} | loss={metrics['loss']:.4f} | {elapsed:.2f}s | {throughput:.1f} tokens/sec")
                # Update progress bar
                progress_bar.update(args.log_every)
                
            # Save checkpoint
            if args.save_path and step % args.save_every == 0 and is_main_process:
                ckpt_path = os.path.join(args.save_path, f"checkpoint_step_{step}.flax")
                print(f"Saving checkpoint to {ckpt_path}")
                save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        elif use_multi_host:
            # Multi-host TPU path
            batch = tree_map(lambda x: jnp.asarray(x), batch)
            
            # Make sure batch size is divisible by total devices
            batch_size = batch["input_ids"].shape[0] 
            seq_len = batch["input_ids"].shape[1]
            
            # For consistent RNG across hosts
            step_rng = jr.fold_in(jr.PRNGKey(global_seed), step)
            
            # Run training step within device mesh context
            with device_mesh:
                start_time = time.time()
                state, loss, metrics = pjit_train_step(state, batch, step_rng)
                elapsed = time.time() - start_time
            
            # Calculate throughput (tokens processed per second)
            batch_tokens = batch_size * seq_len
            total_tokens += batch_tokens
            total_time += elapsed
            
            # Log metrics (only on main process)
            if is_main_process and step % args.log_every == 0:
                # Since we're using pmean, each process should have the same loss value
                throughput = batch_tokens * process_count / max(0.1, elapsed)  # tokens per second across all processes
                print(f"Step {step} | loss={metrics['loss']:.4f} | {elapsed:.2f}s | {throughput:.1f} tokens/sec (global)")
                # Update progress bar
                progress_bar.update(args.log_every)
            
            # Save checkpoint (only on main process)
            if args.save_path and step % args.save_every == 0 and is_main_process:
                ckpt_path = os.path.join(args.save_path, f"checkpoint_step_{step}.flax")
                print(f"Saving checkpoint to {ckpt_path}")
                # No need to extract params from a specific device as we're using pjit
                save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        else:
            # Single-host, multi-device path with pmap
            device_count = jax.device_count()
            # For TPU we need a single RNG key
            step_rng = jr.fold_in(step_rng, step)  # Ensure different key each step
            step_rngs = jax.device_put_replicated(step_rng, jax.devices())
            
            # Convert batch to JAX arrays
            batch = tree_map(lambda x: jnp.asarray(x), batch)
            
            # Make sure batch size is properly divisible by device count
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]
            
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
            
            if is_main_process and step % args.log_every == 0:
                loss_val = metrics["loss"][0] if isinstance(metrics["loss"], jnp.ndarray) else metrics["loss"]
                throughput = batch_tokens / max(0.1, elapsed)  # tokens per second
                print(f"Step {step} | loss={loss_val:.4f} | {elapsed:.2f}s | {throughput:.1f} tokens/sec")
                # Update progress bar
                progress_bar.update(args.log_every)
            
            # Save checkpoint (using only first device's params)
            if args.save_path and step % args.save_every == 0 and is_main_process:
                # Extract params from first device
                params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.params))
                ema_params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
                ckpt_path = os.path.join(args.save_path, f"checkpoint_step_{step}.flax")
                print(f"Saving checkpoint to {ckpt_path}")
                save_checkpoint({"params": params_cpu, "ema_params": ema_params_cpu}, ckpt_path)
        
        if step >= args.steps:
            break
    
    # Close progress bar
    if is_main_process:
        progress_bar.close()
    
    # Save final checkpoint
    if args.save_path and is_main_process:
        if use_cpu:
            ckpt_path = os.path.join(args.save_path, "checkpoint_final.flax")
            save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        elif use_multi_host:
            # For multi-host, we already have the global state
            ckpt_path = os.path.join(args.save_path, "checkpoint_final.flax")
            save_checkpoint({"params": state.params, "ema_params": state.ema_params}, ckpt_path)
        else:
            # Extract params from first device
            params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.params))
            ema_params_cpu = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
            ckpt_path = os.path.join(args.save_path, "checkpoint_final.flax")
            save_checkpoint({"params": params_cpu, "ema_params": ema_params_cpu}, ckpt_path)
    
    # Generate samples if requested (only on main process)
    if args.sample and is_main_process:
        print("\nGenerating text samples...")
        if use_cpu:
            params_to_use = state.ema_params  # Use EMA params for sampling
        elif use_multi_host:
            # For multi-host, we already have the global params
            params_to_use = state.ema_params
        else:
            # Extract params from first device
            params_to_use = jax.device_get(jax.tree.map(lambda x: x[0], state.ema_params))
        
        samples = generate_samples(model, params_to_use, cfg, noise_schedule, num_samples=2, is_multi_host=use_multi_host)
        
        # Print samples (just the token IDs since we don't have a real tokenizer)
        print("Generated token sequences:")
        for i, sample in enumerate(samples):
            print(f"Sample {i+1}: {sample}")
    
    # Print final stats (only on main process)
    if total_time > 0 and is_main_process:
        if use_multi_host:
            # For multi-host, calculate global tokens processed
            global_tokens = total_tokens * process_count
            avg_throughput = global_tokens / total_time
            print(f"\nTraining complete. Average global throughput: {avg_throughput:.1f} tokens/sec across all processes")
        else:
            avg_throughput = total_tokens / total_time
            print(f"\nTraining complete. Average throughput: {avg_throughput:.1f} tokens/sec")


if __name__ == "__main__":
    main() 