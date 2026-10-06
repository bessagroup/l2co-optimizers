"""
Scoring models through a loss on one batch.

:func:`evaluate` is what :func:`~l2co_optimizers.reset` uses to pick the
best of a freshly sampled population, and what l2co's plotting helpers
use to draw loss landscapes. It takes a model and a loss function, not
an optimizer.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.batching import BatchState

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "evaluate",
]


def _evaluate_one(
    model: PyTree,
    batch_state: BatchState,
    dataset: dict[str, jax.Array],
    loss_fn: Callable,
    pass_rng: bool,
    batch_key: PRNGKeyArray,
    loss_key: PRNGKeyArray,
) -> jax.Array:
    """Evaluate a single model on a batch of data.

    Parameters
    ----------
    model : PyTree
        Model to evaluate.
    batch_state : BatchState
        Batch state for data sampling.
    dataset : dict[str, jax.Array]
        Full dataset.
    loss_fn : Callable
        Loss function.
    pass_rng : bool
        Whether to pass RNG key to the loss function.
    batch_key : PRNGKeyArray
        PRNG key for batch sampling.
    loss_key : PRNGKeyArray
        PRNG key for the loss function.

    Returns
    -------
    jax.Array
        Loss value.
    """
    batch_idxs, _ = batch_state.next(batch_key)
    sample = jax.tree.map(lambda x: x[batch_idxs], dataset)
    if pass_rng:
        return loss_fn(model, key=loss_key, **sample)
    else:
        return loss_fn(model, **sample)


evaluate = eqx.filter_vmap(
    _evaluate_one, in_axes=(0, None, None, None, None, None, 0)
)
