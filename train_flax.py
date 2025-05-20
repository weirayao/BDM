from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.experimental import maps  # type: ignore

from models_flax import BD3LMConfig, BD3LMFlax, get_noise_schedule
from models_flax.train_state import TrainState
from models_flax.losses import diffusion_loss

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
# Training step (pmap-ed)
# -----------------------------------------------------------------------------

def create_train_step(cfg: BD3LMConfig, noise_schedule):

    def train_step(state: TrainState, batch, rng):
        loss_fn = lambda params: diffusion_loss(
            cfg, model, params, noise_schedule, batch, rng
        )
        (loss, metrics, new_mutables), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        # Gradient aggregation across devices
        grads = jax.lax.pmean(grads, axis_name="data")
        loss = jax.lax.pmean(loss, axis_name="data")
        metrics = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="data"), metrics)
        state = state.apply_gradients(grads=grads, mutable_state=new_mutables)
        return state, loss, metrics

    model = BD3LMFlax(cfg)
    return jax.pmap(train_step, axis_name="data", donate_argnums=(0,))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--ema", type=float, default=0.9999)
    args = parser.parse_args()

    cfg = BD3LMConfig(model_length=args.seq_len)
    noise_schedule = get_noise_schedule("loglinear")

    model = BD3LMFlax(cfg)
    rng = jr.PRNGKey(42)
    params = model.init(rng, jnp.ones((1, cfg.model_length), jnp.int32), timesteps=jnp.ones((1, 1)))['params']

    tx = optax.adamw(learning_rate=args.learning_rate, b1=0.9, b2=0.95, eps=1e-8)
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, ema_decay=args.ema
    )

    p_train_step = create_train_step(cfg, noise_schedule)

    # replicate across devices
    state = jax.device_put_replicated(state, jax.devices())

    data_iter = synthetic_dataset(cfg.vocab_size, cfg.model_length, args.batch_size, args.steps)

    for step, batch in enumerate(data_iter, 1):
        rng, step_rng = jr.split(rng)
        step_rngs = jr.split(step_rng, jax.local_device_count())
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        batch = jax.device_put_replicated(batch, jax.devices())

        start_time = time.time()
        state, loss, metrics = p_train_step(state, batch, step_rngs)
        if jax.process_index() == 0 and step % 10 == 0:
            print(f"Step {step} | loss={metrics['loss'][0]:.4f} | {time.time() - start_time:.2f}s")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main() 