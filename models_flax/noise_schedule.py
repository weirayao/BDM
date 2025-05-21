from __future__ import annotations

"""Noise schedules in JAX for BD3LM Flax training.

This is a near-line-for-line port of the PyTorch implementation
in `noise_schedule.py`, but rewritten with JAX / Flax idioms.
Each schedule is a simple callable pytree with

    loss_scaling, move_chance = schedule(t)

where `t` is a float32 JAX array in the range \[0, 1].  The
semantics match the original code exactly so that training
hyper-parameters remain compatible.
"""

from typing import Tuple

import jax.numpy as jnp
import flax.struct
import numpy as np

###############################################################################
# Factory helper                                                              #
###############################################################################


def get_noise(noise_type: str = "loglinear"):
    if noise_type == "loglinear":
        return LogLinearNoise()
    elif noise_type == "square":
        return ExpNoise(exp=2.0)
    elif noise_type == "square_root":
        return ExpNoise(exp=0.5)
    elif noise_type == "log":
        return LogarithmicNoise()
    elif noise_type == "cosine":
        return CosineNoise()
    else:
        raise ValueError(f"{noise_type} is not a valid noise schedule")


###############################################################################
# Base class                                                                  #
###############################################################################


@flax.struct.dataclass
class NoiseSchedule:
    """Abstract base class – implemented as dataclass for pytree compliance."""

    def __call__(self, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return (loss_scaling, move_chance) for timestep `t`."""
        raise NotImplementedError()

    # Optional helpers – only implemented for schedules that need them.
    def total_noise(self, t: jnp.ndarray) -> jnp.ndarray:  # pylint: disable=unused-argument
        raise NotImplementedError()

    def rate_noise(self, t: jnp.ndarray) -> jnp.ndarray:  # pylint: disable=unused-argument
        raise NotImplementedError()


###############################################################################
# Concrete schedules                                                          #
###############################################################################


@flax.struct.dataclass
class CosineNoise(NoiseSchedule):
    eps: float = 1e-3

    def __call__(self, t: jnp.ndarray):
        cos = -(1 - self.eps) * jnp.cos(t * jnp.pi / 2)
        sin = -(1 - self.eps) * jnp.sin(t * jnp.pi / 2)
        move_chance = cos + 1.0
        loss_scaling = sin / (move_chance + self.eps) * (jnp.pi / 2)
        return loss_scaling, move_chance


@flax.struct.dataclass
class ExpNoise(NoiseSchedule):
    exp: float = 2.0  # exponent
    eps: float = 1e-3

    def __call__(self, t: jnp.ndarray):
        move_chance = jnp.clip(jnp.power(t, self.exp), a_min=self.eps)
        loss_scaling = - (self.exp * jnp.power(t, self.exp - 1.0)) / move_chance
        return loss_scaling, move_chance


@flax.struct.dataclass
class LogarithmicNoise(NoiseSchedule):
    eps: float = 1e-3

    def __call__(self, t: jnp.ndarray):
        ln2 = jnp.log(2.0)
        move_chance = jnp.log1p(t) / ln2
        loss_scaling = -1.0 / (move_chance * ln2 * (1.0 + t))
        return loss_scaling, move_chance


@flax.struct.dataclass
class LogLinearNoise(NoiseSchedule):
    eps: float = 1e-3

    # ---------------------------------------------------------------------
    # Convenience wrappers matching PyTorch implementation used in Diffusion
    # ---------------------------------------------------------------------
    def total_noise(self, t: jnp.ndarray) -> jnp.ndarray:
        return -jnp.log1p(-(1.0 - self.eps) * t)

    def rate_noise(self, t: jnp.ndarray) -> jnp.ndarray:
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)

    # ---------------------------------------------------------------------
    def __call__(self, t: jnp.ndarray):
        # loss_scaling and move_chance as in original repo
        loss_scaling = -1.0 / t
        move_chance = t
        return loss_scaling, move_chance

    # sigma range helpers (hardcoded for JAX tracing compatibility)
    @property
    def sigma_max(self) -> float:
        # Hardcoded value for eps=1e-3: -log(1-(1-0.001)*1) = -log(0.001) ~= 6.9
        return 6.9078

    @property
    def sigma_min(self) -> float:
        # eps + total_noise(0.0) = eps + 0
        return self.eps 