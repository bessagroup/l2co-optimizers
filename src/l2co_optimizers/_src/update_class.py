"""
The ``UpdateClass`` adapter.

``UpdateClass`` is the static, JIT-friendly container that the rollout
machinery (``init_``, ``step``, ``run_``, ``batch_run_``) consumes. This
module holds only the container; the factories that build one from a
``Task`` plus an optimizer specification live with their library --
``optax_update`` / ``optax_update_extra_kwargs`` in
:mod:`~l2co_optimizers._src.optax_implementations`,
``evosax_distribution_update`` / ``evosax_population_update`` in
:mod:`~l2co_optimizers._src.evosax_implementations`, and the other
built-ins (``lbfgs_update``, ``shade_update``, ``random_search_update``,
...) in their own modules.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from typing import ClassVar

# Third-party
import equinox as eqx

# Local
from l2co_optimizers._src.state_transfer import (
    FAMILY_GRADIENT,
)
from l2co_optimizers._src.typing import (
    AskFunction,
    InitFunction,
    StepFunction,
    StopFunction,
    TransferReadFunction,
    TransferWriteFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


#                                                                   UpdateClass
# =============================================================================


# =============================================================================


class UpdateClass(eqx.Module):
    """Static container wrapping one optimizer step.

    Encapsulates the initialization function, step function, population
    size, optimizer hash, and stopping criterion for a single optimizer.
    Constructed via a factory function such as ``optax_update`` /
    ``evosax_*_update`` (or a subclass like ``RandomSearchUpdateClass``); the
    instance is then handed to the rollout helpers in
    ``l2co._src.update_class``.

    Attributes
    ----------
    init_fn : InitFunction
        Function to initialize optimizer state.
    step_fn : StepFunction
        Function to perform a single optimization step.
    popsize : int
        Population size for the optimizer (1 for gradient-based).
    hash : int
        Stable hash identifier for the optimizer configuration.
    stop_fn : StopFunction or None
        ``stop_fn(recent_history, opt_state) -> bool`` -- early-stopping
        check, evaluated before every step. ``None`` (default) never
        stops, and is resolved at trace time in
        :func:`l2co._src.update_class.step`, so it costs nothing.
    max_fevals_per_step : int or None
        Most function evaluations this optimizer can bill in one
        iteration, or ``None`` (default) when that is just ``popsize``.
        Only linesearch optimizers set it: L-BFGS bills one evaluation
        per linesearch trial, so it can spend far more than its
        population of one (see :func:`step_fevals` and
        ``l2co ADR 0010``). Read it through :attr:`fevals_bound`, which
        resolves the ``None``. Consumers that size a budget or
        normalize a per-evaluation cost must use that bound rather than
        ``popsize`` -- notably rl2co's ``AUCEnv``, whose reward
        normalization assumes no optimizer bills more than
        ``max_iterations * max(popsize)``.
    family : str
        Which optimizer family this is, one of
        :data:`~l2co_optimizers._src.state_transfer.FAMILIES`. Set by
        the constructing factory, which knows what it is building --
        *not* inferred from evosax registry membership, which
        ``FEEDBACK_evosax.md`` records as a source of friction. Says
        what kind of search state this optimizer holds, and nothing
        more: the population half of a handshake is decided by
        :attr:`ask_fn`'s presence, not by this.
    ask_fn : AskFunction or None
        ``ask_fn(opt_state, key) -> (params, opt_state)`` -- the
        optimizer's own sampling step, bounds-clipped like
        :attr:`step_fn`. Present when
        :attr:`~l2co_optimizers._src.state_transfer.TransferSpec.own_ask`
        is set, which is every ``"distribution"`` optimizer and the
        mutation GAs; ``None`` (default) otherwise. A switching caller
        that finds it set must draw the post-switch population from it
        rather than handing over a layout. Calling it *after* writing
        the handshake into the state is what keeps the optimizer's next
        update consistent with the population it is told about:
        CR-FM-NES and seven relatives cache their draws in state and
        consume them on the next ``tell`` (``l2co ADR 0012``), and the
        mutation GAs score the next generation against a baseline the
        writer just repaired (``l2co ADR 0014``).
    transfer_read_fn : TransferReadFunction or None
        ``transfer_read_fn(opt_state) -> (sigma, conf)`` -- projects
        this optimizer's own state onto the canonical scale carried
        across a switch. Always callable when the optimizer was built
        through a factory: an optimizer holding no scale gets a reader
        returning ``CONF_ABSENT`` rather than ``None``, so callers need
        no presence check. ``None`` only for a hand-constructed
        ``UpdateClass``.
    transfer_write_fn : TransferWriteFunction or None
        ``transfer_write_fn(bundle, opt_state) -> opt_state`` -- folds a
        :class:`~l2co_optimizers._src.state_transfer.TransferBundle`
        into this optimizer's own state. Same convention: a
        non-receiver such as L-BFGS gets
        :func:`~l2co_optimizers._src.state_transfer.write_none`, an
        identity, rather than ``None``. Both halves are bound at
        construction because they need hyperparameters the state does
        not retain -- the learning rate, Rprop's clip bounds, the ravel
        function evosax uses to flatten a solution.
    sequential_realizations : bool
        Class-level marker (not a field): whether a multi-realization
        run must execute one realization at a time rather than with all
        realizations vectorized together. ``False`` for every plain
        optimizer, which is the fast choice. A subclass sets it to
        ``True`` when its ``step_fn`` branches on a value that differs
        per realization -- the meta-optimizers, whose step picks a
        sub-optimizer from a menu. Read by
        :func:`l2co._src.run_state.batch_run`, which is where the
        consequences are spelled out.
    """

    #: See the class docstring. A ``ClassVar`` annotation keeps this out
    #: of the dataclass fields, so subclasses override it with a plain
    #: assignment and no constructor signature changes.
    sequential_realizations: ClassVar[bool] = False

    init_fn: InitFunction = eqx.field(static=True)
    step_fn: StepFunction = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    hash: int = eqx.field(static=True)
    stop_fn: StopFunction | None = eqx.field(static=True, default=None)
    max_fevals_per_step: int | None = eqx.field(static=True, default=None)
    family: str = eqx.field(static=True, default=FAMILY_GRADIENT)
    ask_fn: AskFunction | None = eqx.field(static=True, default=None)
    transfer_read_fn: TransferReadFunction | None = eqx.field(
        static=True, default=None
    )
    transfer_write_fn: TransferWriteFunction | None = eqx.field(
        static=True, default=None
    )

    @property
    def fevals_bound(self) -> int:
        """Upper bound on ``OptHistory.fevals`` for one iteration.

        Returns
        -------
        int
            :attr:`max_fevals_per_step` when set, else :attr:`popsize`.
        """
        if self.max_fevals_per_step is None:
            return self.popsize
        return self.max_fevals_per_step
