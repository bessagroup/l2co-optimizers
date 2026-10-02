"""Population loss evaluators.

Evaluate a task's loss -- and optionally its gradient -- over a leading
population axis of model parameters. ``sample`` is the current data
batch, forwarded to ``loss_fn`` as keyword arguments; the ``_with_rng``
variants also pass a per-candidate ``key`` for stochastic losses.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def combined_loss(model: PyTree, loss_fn: Callable, sample: dict) -> jax.Array:
    """
    Compute the combined loss for a given set of parameters.

    Parameters
    ----------
    model : PyTree
        The model parameters.
    loss_fn : Callable
        The loss function to compute the loss.
    sample : dict
        Sample data for loss computation.

    Returns
    -------
    jax.Array
        The computed loss.
    """
    return loss_fn(model, **sample)


def combined_loss_with_rng(
    model: PyTree, loss_fn: Callable, sample: dict, key: PRNGKeyArray
) -> jax.Array:
    """
    Compute the combined loss for a given set of parameters with RNG.

    Parameters
    ----------
    model : PyTree
        The model parameters.
    loss_fn : Callable
        The loss function to compute the loss.
    sample : dict
        Sample data for loss computation.
    key : PRNGKeyArray
        Random number generator key.

    Returns
    -------
    jax.Array
        The computed loss.
    """
    return loss_fn(model, **sample, key=key)


vmapped_loss_and_grad = eqx.filter_vmap(
    eqx.filter_value_and_grad(combined_loss), in_axes=(0, None, None)
)
vmapped_loss = eqx.filter_vmap(combined_loss, in_axes=(0, None, None))


vmapped_loss_and_grad_with_rng = eqx.filter_vmap(
    eqx.filter_value_and_grad(combined_loss_with_rng),
    in_axes=(0, None, None, 0),
)

vmapped_loss_with_rng = eqx.filter_vmap(
    combined_loss_with_rng,
    in_axes=(0, None, None, 0),
)
