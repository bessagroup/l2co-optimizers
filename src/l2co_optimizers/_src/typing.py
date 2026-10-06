"""
Type aliases and protocols forming the optimizer contract.

These are the types an optimizer author implements against: what a
step consumes and returns, what a stopping check sees, and
:class:`TaskLike`, the only view of a task this package takes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from abc import abstractmethod

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx

# Third-party
from evosax.algorithms.base import State as EvoSaxState
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from optax import OptState as OptaxOptState

if TYPE_CHECKING:
    from l2co_optimizers._src.batching import BatchState
    from l2co_optimizers._src.opt_history import RecentHistory
    from l2co_optimizers._src.state_transfer import TransferBundle

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class OptHistoryType(eqx.Module):
    """Abstract base class for optimization history.

    Defines the interface and required fields for storing
    per-step losses, parameters, gradients, and evaluation
    counters during optimization.
    """

    loss: eqx.AbstractVar[Float[Array, " popsize"]]
    params: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    grads: eqx.AbstractVar[Float[PyTree, " popsize dim"]]
    update_step: eqx.AbstractVar[Int[Array, " popsize"]]
    fevals: eqx.AbstractVar[Int[Array, " popsize"]]
    iterations: eqx.AbstractVar[Int[Array, " popsize"]]

    @classmethod
    @abstractmethod
    def init(
        cls, params: PyTree, popsize: int, last_step: int
    ) -> OptHistoryType:
        """Initialize the history."""
        raise NotImplementedError

    @abstractmethod
    def flatten(self) -> OptHistoryType:
        """
        Flatten the first and second dimension of the loss,
        params and grad.
        """
        raise NotImplementedError


# =============================================================================

InputParameters = Float[PyTree, " popsize dimensionality"]
OptState = PyTree

#: ``step_fn(carry, sample) -> (carry, history)`` -- one optimizer
#: generation. ``carry`` is ``(params, opt_state, key)``; ``sample`` is the
#: current data batch, forwarded to the loss as keyword arguments.
StepFunction = Callable[
    [tuple[InputParameters, OptState, PRNGKeyArray], dict[str, Array]],
    tuple[tuple[InputParameters, OptState, PRNGKeyArray], OptHistoryType],
]

#: ``stop_fn(recent_history, opt_state) -> stop`` -- early-stopping check
#: evaluated before every step, on the rolling ``RecentHistory`` window.
StopFunction = Callable[["RecentHistory", OptState], Bool[Array, ""]]

#: ``init_fn(params, key=None, **sample) -> opt_state``. ``Callable[...]``
#: because the batch arrives as keyword arguments, which ``Callable``
#: cannot spell.
InitFunction = Callable[..., OptaxOptState | EvoSaxState]

#: ``ask_fn(opt_state, key) -> (params, opt_state)`` -- an optimizer's own
#: sampling step, used by a handshake to draw the post-switch population.
AskFunction = Callable[
    [OptState, PRNGKeyArray], tuple[InputParameters, OptState]
]

#: ``transfer_read_fn(opt_state) -> (sigma, conf_sigma)`` -- projects an
#: optimizer's state onto the scale carried across a switch.
TransferReadFunction = Callable[
    [OptState], tuple[Float[Array, ""], Float[Array, ""]]
]

#: ``transfer_write_fn(bundle, opt_state) -> opt_state`` -- folds a
#: ``TransferBundle`` into an optimizer's own state.
TransferWriteFunction = Callable[["TransferBundle", OptState], OptState]

#: ``loss_fn(model, **sample) -> loss`` for one (unbatched) model, or
#: ``loss_fn(model, key=key, **sample)`` when ``task.pass_rng``. Scalar
#: output; the ``vmapped_loss*`` helpers add the population axis.
#: ``Callable[...]`` because the batch arrives as keyword arguments.
LossFunction = Callable[..., Float[Array, ""]]


@runtime_checkable
class TaskLike(Protocol):
    """The view of a task an optimizer factory takes.

    Structural: any object with these three attributes qualifies, so
    ``l2co_tasks.Task`` satisfies it without either package importing
    the other. Meta-optimizer factories registered into the same
    registry may require more (a full ``Task``); the bare optimizers
    read only these.

    Attributes
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : Callable
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when :attr:`pass_rng` -- returning a scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    """

    model: PyTree
    loss_fn: Callable
    pass_rng: bool


#: Population size: a literal ``int`` or a callable deriving one from the
#: task (see :mod:`l2co_optimizers._src.popsize`).
PopSize = int | Callable[[TaskLike], int]

SamplerFunction = Callable[[PRNGKeyArray, PyTree, int], InputParameters]

#: The scan carry of :meth:`UpdateClass.run`: ``(params, opt_state, key,
#: batch_state, done, recent_history)``. The first three are what
#: ``step_fn`` sees; the rest is the loop's own bookkeeping.
Carry = tuple[
    InputParameters,
    OptState,
    PRNGKeyArray,
    "BatchState",
    Bool[Array, ""],
    "RecentHistory",
]
