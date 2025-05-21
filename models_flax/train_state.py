from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import flax.struct
import optax
from flax.training import train_state as ts


@flax.struct.dataclass
class TrainState(ts.TrainState):
    """Enhanced TrainState that keeps EMA shadow parameters and extra mutables.

    Attributes
    ----------
    ema_params : pytree
        Exponential-moving-average of the model parameters.  Updated every
        step with decay `ema_decay`.
    mutable_state : pytree | None
        Collection of mutable variables (e.g., BatchNorm stats, KV cache) that
        should be threaded through `apply` but not optimised.
    """

    ema_params: Any
    ema_decay: float = flax.struct.field(pytree_node=False)
    mutable_state: Optional[Any] = None

    @classmethod
    def create(
        cls,
        *,
        apply_fn: Callable[..., Any],
        params: Any,
        tx: optax.GradientTransformation,
        ema_decay: float,
    ) -> "TrainState":
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=params,
            tx=tx,
            opt_state=tx.init(params),
            mutable_state=None,
            ema_decay=ema_decay,
        )

    # ------------------------------------------------------------------
    # Flax `TrainState` API overrides
    # ------------------------------------------------------------------
    def apply_gradients(self, *, grads: Any, mutable_state: Any | None = None):
        new_state = super().apply_gradients(grads=grads)
        new_ema_params = optax.incremental_update(
            new_state.params, self.ema_params, step_size=1.0 - self.ema_decay
        )
        return new_state.replace(ema_params=new_ema_params, mutable_state=mutable_state) 