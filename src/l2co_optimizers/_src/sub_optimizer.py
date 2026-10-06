"""Sub-optimizer adapters shared by the strategy wrappers.

A :class:`SubOpt` is a thin static adapter that hides whether a
sub-optimizer is gradient-based (an optax ``GradientTransformation``)
or population-based (an evosax-style ``(init, ask, tell)`` triple).
Wrappers that expose a selection model as a duck-typed evosax
strategy — l2co's own :mod:`l2co._src.strategy_wrapper` and rl2co's
policy-deployment wrappers (``rl2co._src.optax_wrapper`` /
``rl2co._src.evosax_wrapper``, which import this module through a
shim) — dispatch across a tuple of ``SubOpt`` adapters via
:func:`jax.lax.switch`, reading the static ``is_pop`` flag at Python
trace time to pick which branch implementation to build for each
slot.

Two factory functions build ``SubOpt`` instances directly
(:func:`grad_sub_optimizer`, :func:`pop_sub_optimizer`); a third,
:func:`optstep_to_subopt`, dispatches an
:class:`~l2co_optimizers._src.optimizer_schedule.OptimizationStep` to the
appropriate factory by looking the optimizer name up in the canonical
l2co registries.

The module was authored in rl2co and promoted here because it imports
only l2co internals and both packages' wrappers need it — see
``l2co ADR 0003-promote-subopt-adapter-from-rl2co.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from l2co_optimizers._src.evosax_implementations import normalized_evosax
from l2co_optimizers._src.handshake_policy import (
    POPULATION_HANDSHAKES,
    handshake_policy_for,
)
from l2co_optimizers._src.lbfgs import (
    DEFAULT_MAX_LINESEARCH_STEPS,
    stock_lbfgs,
)
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
    step_fevals,
)
from l2co_optimizers._src.optimizer_schedule import OptimizationStep
from l2co_optimizers._src.rbf_trust_region import RBFTrustRegion
from l2co_optimizers._src.shade import SHADE
from l2co_optimizers._src.state_transfer import (
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    build_transfer_fns,
    transfer_spec_for,
)
from l2co_optimizers._src.turbo import TuRBO
from l2co_optimizers._src.typing import TaskLike
from l2co_optimizers._src.utils import normalize_key

# =============================================================================

__all__ = [
    "CONSTRUCTOR_HYPERPARAMETERS",
    "LINESEARCH_FEVAL_BOUND",
    "SubOpt",
    "grad_sub_optimizer",
    "l2co_native_evosax",
    "l2co_native_optax",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "resolve_popsize",
]


# l2co-native evosax-style algorithms: absent from evosax's own
# registries but subclassing its base API, so the generic evosax branch
# of ``optstep_to_subopt`` drives them unchanged.
# Keys are the normalized names (``normalize_key``), which is what
# ``optstep_to_subopt`` dispatches on -- an underscored name registered
# raw resolves directly but not as a sub-step.
l2co_native_evosax: dict[str, type] = {
    "shade": SHADE,
    "rbftrustregion": RBFTrustRegion,
    "turbo": TuRBO,
}

# l2co-native optax-style transforms: assembled by l2co rather than
# named on the ``optax`` module, so they are absent from the
# ``normalized_optax_*`` registries but the generic gradient branch of
# ``optstep_to_subopt`` drives them unchanged. ``"lbfgs"`` lives here
# because it moved out of ``opt_names_from_state`` into its own module
# (``l2co_optimizers._src.lbfgs``) to gain per-eval-key semantics; the
# ``SubOpt`` path takes the stock variant, which is the only one it can
# drive (see :func:`optstep_to_subopt`).
l2co_native_optax: dict[str, Callable[..., optax.GradientTransformation]] = {
    "lbfgs": stock_lbfgs,
}

# Gradient transforms that spend more than one evaluation per step, keyed
# as in the registry, mapping their hyperparameters to the worst case.
# A line search evaluates the objective once per trial, so its cost is
# unrelated to its population of one -- consumers that size a budget or
# normalise a per-evaluation quantity must use ``SubOpt.fevals_bound``
# rather than ``popsize`` (see ``l2co ADR 0010``). Mirrors what
# ``UpdateClass.max_fevals_per_step`` carries on the registry path.
LINESEARCH_FEVAL_BOUND: dict[str, Callable[[dict], int]] = {
    "lbfgs": lambda hp: (
        int(hp.get("max_linesearch_steps", DEFAULT_MAX_LINESEARCH_STEPS)) + 1
    ),
}

# Shape-affecting hyperparameter names consumed by an algorithm's
# *constructor* rather than its flax-struct ``Params`` (which only
# holds value-like leaves). These are stripped from the hyperparameters
# before the ``default_params.replace(...)`` overrides.
CONSTRUCTOR_HYPERPARAMETERS: dict[str, tuple[str, ...]] = {
    "shade": ("memory_size", "archive_size"),
    "rbftrustregion": ("archive_size", "n_candidates"),
    "turbo": ("archive_size", "n_candidates"),
}


#                                                       Sub-optimizer adapter
# =============================================================================


class SubOpt(eqx.Module):
    """Internal adapter for one sub-optimizer.

    Every entry of a wrapper's sub-optimizer tuple is a ``SubOpt``;
    the wrapper dispatches across them via :func:`jax.lax.switch` and
    reads the static ``is_pop`` flag at Python trace time to decide
    which branch implementation to build for each slot. All fields
    are static — a ``SubOpt`` contributes no array leaves to the
    surrounding pytree.

    Build via :func:`grad_sub_optimizer` or :func:`pop_sub_optimizer`,
    or have the wrapper factories build them automatically from
    :class:`~l2co_optimizers._src.optimizer_schedule.OptimizationStep` entries
    via :func:`optstep_to_subopt`.

    Attributes
    ----------
    is_pop : bool
        ``True`` for population-based (evosax-style) adapters,
        ``False`` for gradient-based (optax-style) adapters. Read at
        Python trace time to pick the ``lax.switch`` branch
        implementation for each sub-optimizer slot.
    popsize : int
        Population size. ``1`` for gradient-based.
    name_hash : int
        Identifier written into ``HistoryState.update_step`` so the
        selection model can tell which optimizer produced each entry.
    init_fn : Callable
        ``init_fn(params, key) -> sub_state``.
    step_fn : Callable
        Gradient-based: ``(grads, params, sub_state, key, **extra_args)
        -> (update, new_sub_state)`` — returns the optax-style negated
        update. ``**extra_args`` are forwarded verbatim to the optax
        transform's ``update``; line-search transforms (e.g.
        ``optax.lbfgs``) require ``value`` / ``grad`` / ``value_fn``
        there, while plain transforms (Adam, SGD, …) ignore them.
        Population-based: ``(params, sub_state, key) -> (population,
        new_sub_state)`` — i.e. an ``ask`` call returning a fresh
        population of shape ``(popsize, *params_shape)``.
    tell_fn : Callable | None
        Population-based only: ``(sub_state, population, fitnesses,
        key) -> new_sub_state``. ``None`` for gradient-based.
    opt_state_handshake : str
        ``"continue"`` (default) or ``"reset"`` — whether this
        sub-optimizer's internal state survives a handshake, or must be
        re-initialised around the best parameters because it cannot.
        Resolved from :data:`~l2co_optimizers._src.handshake_policy.
        HANDSHAKE_POLICY` by :func:`optstep_to_subopt`; read at Python
        trace time by the strategy wrappers, so a ``'continue'``
        sub-optimizer's reset branch is never traced. See
        ``l2co ADR 0009-shared-handshake-policy-table.md``.
    population_handshake : str
        ``"best_one"`` (default) or ``"best"`` -- which population the
        incoming optimizer sees. Resolved from the same shared table as
        :attr:`opt_state_handshake` by :func:`optstep_to_subopt`. Apply it
        through :meth:`handshake_population`, never by reimplementing the
        layout: the slot the best point lands in changes the update a
        mirrored-sampling algorithm computes.
    family : str
        Which optimizer family this is, one of
        :data:`~l2co_optimizers._src.state_transfer.FAMILIES`. Says what
        kind of search state this sub-optimizer holds, and nothing more
        -- the population half of a switch is decided by
        :attr:`own_ask`.
    own_ask : bool
        Whether this sub-optimizer regenerates its own post-switch
        population. Read at Python trace time by the strategy wrappers:
        where it is set they call :attr:`step_fn` (which *is* the
        ``ask`` on the population path) against the state a handshake
        just wrote, rather than handing over a layout. That is what
        keeps an optimizer's next update consistent with the population
        it is told about -- the ask/tell-coupled evosax algorithms cache
        their draws in state (``l2co ADR 0012``), and the mutation GAs
        score the next generation against a baseline the writer just
        repaired (``l2co ADR 0014``). Resolved from the same
        :func:`~l2co_optimizers._src.state_transfer.transfer_spec_for`
        table the ``UpdateClass`` path reads, so the environment and
        both deployment wrappers agree.
    transfer_read_fn : Callable | None
        ``transfer_read_fn(sub_state) -> (sigma, conf)`` -- projects this
        sub-optimizer's own state onto the scale a switch carries.
        Resolved from the same
        :func:`~l2co_optimizers._src.state_transfer.build_transfer_fns`
        the ``UpdateClass`` path uses, so the environment and both
        deployment wrappers agree (``l2co ADR 0009``'s constraint,
        applied to ``l2co ADR 0013``'s channel).
    transfer_write_fn : Callable | None
        ``transfer_write_fn(bundle, sub_state) -> sub_state`` -- folds a
        carried scale into this sub-optimizer's own state.
    fevals_fn : Callable | None
        ``fevals_fn(sub_state_before, sub_state_after) -> Array`` — the
        function evaluations one step of this sub-optimizer actually
        spent. ``None`` (the default, and the case for every
        population-based adapter) means the step costs exactly
        :attr:`popsize`. :func:`grad_sub_optimizer` wires
        :func:`~l2co_optimizers._src.optax_implementations.step_fevals`
        in here, which is what makes a *line-search* transform bill its true
        trial count instead of a flat ``1``. Read it through
        :meth:`bill_fevals`.
    """

    is_pop: bool = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    name_hash: int = eqx.field(static=True)
    init_fn: Callable = eqx.field(static=True)
    step_fn: Callable = eqx.field(static=True)
    tell_fn: Callable | None = eqx.field(static=True)
    opt_state_handshake: str = eqx.field(static=True, default="continue")
    fevals_fn: Callable | None = eqx.field(static=True, default=None)
    population_handshake: str = eqx.field(static=True, default="best_one")
    max_fevals_per_step: int | None = eqx.field(static=True, default=None)
    family: str = eqx.field(static=True, default=FAMILY_GRADIENT)
    transfer_read_fn: Callable | None = eqx.field(static=True, default=None)
    transfer_write_fn: Callable | None = eqx.field(static=True, default=None)
    own_ask: bool = eqx.field(static=True, default=False)

    @property
    def fevals_bound(self) -> int:
        """Worst-case function evaluations one step of this can spend.

        :attr:`max_fevals_per_step` when set, else :attr:`popsize`. The two
        differ for line-search transforms, which evaluate the objective
        once per trial against a population of one.

        This is the *bound*, for sizing and normalising; the actual cost of
        a given step comes from :meth:`bill_fevals`. Consumers that
        normalise a per-evaluation quantity (rl2co's ``AUCEnv`` reward and
        its feval-progress observation channel) must use this rather than
        ``popsize``, or they under-estimate the worst case and let a
        normalised quantity leave its intended range.

        Returns
        -------
        int
            Positive upper bound on one step's evaluations.
        """
        if self.max_fevals_per_step is None:
            return self.popsize
        return self.max_fevals_per_step

    def handshake_population(
        self,
        best_params: PyTree,
        sampler: Callable,
        key: PRNGKeyArray,
    ) -> PyTree:
        """Build the population this sub-optimizer receives on a switch.

        Dispatches on :attr:`population_handshake` into
        :data:`~l2co_optimizers._src.handshake_policy.
        POPULATION_HANDSHAKES`, the single canonical implementation the
        rl2co environment also uses -- so a deployed policy sees the same
        post-switch population it was trained against. The returned
        leading axis is :attr:`popsize`; a caller whose generation width
        is wider pads it.

        Parameters
        ----------
        best_params : PyTree
            Best parameters seen so far, as one solution.
        sampler : Callable
            ``l2co.sampling`` function, called as
            ``sampler(key, best_params, popsize)``.
        key : PRNGKeyArray
            Key for the draws.

        Returns
        -------
        PyTree
            Population with a leading axis of :attr:`popsize`.
        """
        return POPULATION_HANDSHAKES[self.population_handshake](
            best_params, self.popsize, sampler, key
        )

    def bill_fevals(
        self, sub_state_before: PyTree, sub_state_after: PyTree
    ) -> Array:
        """Function evaluations one step of this sub-optimizer spent.

        Consumers that record ``OptHistory.fevals`` /
        ``HistoryState.fevals`` from an ask/tell wrapper must bill
        through here rather than assuming :attr:`popsize`. The two
        differ for line-search optimizers: ``lbfgs`` has a population
        of one but evaluates the objective once per line-search trial,
        up to ``max_linesearch_steps + 1`` in a single step (see
        ``l2co ADR 0010-lbfgs-bills-one-feval-per-linesearch-trial.md``).
        Billing ``popsize`` there under-reports its cost by up to 21x
        and makes a wrapper-driven run non-comparable with the
        registry-driven runs it is measured against.

        Because the count is only known *after* the step, a caller that
        records history before dispatching the step has to reorder.

        Parameters
        ----------
        sub_state_before : PyTree
            This sub-optimizer's internal state before the step.
        sub_state_after : PyTree
            Its state after the step, holding the line-search info.

        Returns
        -------
        Array
            Scalar integer feval count for this step.
        """
        if self.fevals_fn is None:
            return jnp.asarray(self.popsize, dtype=int)
        return jnp.asarray(
            self.fevals_fn(sub_state_before, sub_state_after), dtype=int
        )


def grad_sub_optimizer(
    transform: optax.GradientTransformation,
    name_hash: int,
    opt_state_handshake: str = "continue",
    population_handshake: str = "best_one",
    max_fevals_per_step: int | None = None,
    family: str = FAMILY_GRADIENT,
    transfer_read_fn: Callable | None = None,
    transfer_write_fn: Callable | None = None,
) -> SubOpt:
    """Wrap an optax gradient transformation as a ``SubOpt``.

    Hyperparameters live inside ``transform`` (closed over per the
    optax convention; see rl2co's
    ``agent-reviews/optax_gradient_transformation_api.md`` §3).

    Parameters
    ----------
    transform : optax.GradientTransformation
        Any optax transformation. Its ``update`` is called once per
        outer step.
    name_hash : int
        Stable identifier written into ``HistoryState.update_step``;
        must be unique across the sub-optimizer tuple. Typically the
        hash of the source :class:`OptimizationStep`.
    opt_state_handshake : str, optional
        ``"continue"`` (default) or ``"reset"``; see
        :class:`SubOpt`. Defaulted so hand-built adapters keep the
        universal behaviour without opting in.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.
    """

    def init_fn(params: PyTree, key: PRNGKeyArray) -> PyTree:
        """Initialise the optax transform's internal state.

        Parameters
        ----------
        params : PyTree
            Parameters used to shape Adam moments, etc.
        key : PRNGKeyArray
            Unused; accepted for interface uniformity with
            population-based ``init_fn``.

        Returns
        -------
        PyTree
            optax optimizer state.
        """
        del key
        return transform.init(params)

    def step_fn(
        grads: PyTree,
        params: PyTree,
        sub_state: PyTree,
        key: PRNGKeyArray,
        **extra_args: Any,
    ) -> tuple[PyTree, PyTree]:
        """Apply one optax step.

        Parameters
        ----------
        grads : PyTree
            Outer-loop gradients for ``params``.
        params : PyTree
            Current parameters; forwarded to transforms that need
            them (e.g. weight decay).
        sub_state : PyTree
            Previous optax state.
        key : PRNGKeyArray
            Unused; accepted for interface uniformity.
        **extra_args
            Forwarded verbatim to ``transform.update`` as keyword
            arguments. Line-search transforms (e.g. ``optax.lbfgs``,
            which chains ``scale_by_zoom_linesearch``) require
            ``value`` (the current loss), ``grad`` (the gradient at
            ``params``), and ``value_fn`` (a callable ``params ->
            loss`` the line search probes at trial points). Plain
            transforms (Adam, SGD, …) accept and ignore these via
            optax's ``GradientTransformationExtraArgs`` support, so
            passing them is always safe; omitting them leaves the
            stock single-argument behaviour unchanged.

        Returns
        -------
        tuple[PyTree, PyTree]
            ``(updates, new_sub_state)`` — already optax-sign-correct.
        """
        del key
        return transform.update(grads, sub_state, params, **extra_args)

    return SubOpt(
        is_pop=False,
        popsize=1,
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=step_fn,
        tell_fn=None,
        opt_state_handshake=opt_state_handshake,
        population_handshake=population_handshake,
        max_fevals_per_step=max_fevals_per_step,
        # A plain transform bills 1 (== popsize) and a line-search
        # transform its true trial count, from the same helper the
        # registry path uses -- so an ask/tell wrapper's fevals are
        # comparable with a registry-driven run's. See
        # ``SubOpt.bill_fevals``.
        fevals_fn=step_fevals,
        family=family,
        transfer_read_fn=transfer_read_fn,
        transfer_write_fn=transfer_write_fn,
    )


def pop_sub_optimizer(
    init_fn: Callable[[PyTree, PRNGKeyArray], PyTree],
    ask_fn: Callable[[PyTree, PyTree, PRNGKeyArray], tuple[PyTree, PyTree]],
    tell_fn: Callable[[PyTree, PyTree, Array, PRNGKeyArray], PyTree],
    popsize: int,
    name_hash: int,
    opt_state_handshake: str = "continue",
    population_handshake: str = "best_one",
    max_fevals_per_step: int | None = None,
    family: str = FAMILY_GRADIENT,
    transfer_read_fn: Callable | None = None,
    transfer_write_fn: Callable | None = None,
    own_ask: bool = False,
) -> SubOpt:
    """Wrap an evosax-style ``(init, ask, tell)`` triple as a ``SubOpt``.

    The ``init/ask/tell`` callables are supplied by the user so this
    layer does not import evosax directly; see
    :func:`optstep_to_subopt` for the canonical wiring.

    Parameters
    ----------
    init_fn : Callable
        ``(params, key) -> sub_state``. Builds the ES internal state.
    ask_fn : Callable
        ``(params, sub_state, key) -> (population, new_sub_state)``,
        with ``population`` of shape ``(popsize, *params_shape)``.
    tell_fn : Callable
        ``(sub_state, population, fitnesses, key) -> new_sub_state``.
    popsize : int
        Number of candidates emitted per ``ask``. Static.
    name_hash : int
        Stable identifier; see :func:`grad_sub_optimizer`.
    opt_state_handshake : str, optional
        ``"continue"`` (default) or ``"reset"``; see :class:`SubOpt`.
    own_ask : bool, optional
        Whether a switch should draw this sub-optimizer's post-switch
        population from ``ask_fn`` instead of handing it a layout; see
        :class:`SubOpt`. Defaults to False.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.
    """
    return SubOpt(
        is_pop=True,
        popsize=int(popsize),
        name_hash=int(name_hash),
        init_fn=init_fn,
        step_fn=ask_fn,
        tell_fn=tell_fn,
        opt_state_handshake=opt_state_handshake,
        population_handshake=population_handshake,
        max_fevals_per_step=max_fevals_per_step,
        family=family,
        transfer_read_fn=transfer_read_fn,
        transfer_write_fn=transfer_write_fn,
        own_ask=own_ask,
    )


#                                       OptimizationStep -> SubOpt dispatch
# =============================================================================


def resolve_popsize(opt_step: OptimizationStep, task: TaskLike) -> int:
    """Resolve the canonical popsize for ``opt_step`` via l2co's rules.

    Builds an :class:`~l2co_optimizers._src.update_class.UpdateClass`
    purely to read its ``popsize`` attribute — ``1`` for optax-style
    optimizers, an explicit value or ``int(4 + 3 * log(d))`` for
    evosax — so the wrapper's population sizing matches the call site
    in :meth:`~l2co_optimizers.RunState.init`. The ``UpdateClass``
    instance itself is discarded; the wrapper builds its own
    optax/evosax adapter against the same hyperparameters and the
    correctly-resolved popsize.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.hyperparameters`` (including any
        explicit ``popsize`` kwarg) is forwarded to the factory.
    task : TaskLike
        Forwarded to :func:`~l2co_optimizers._src.mapping.
        optimizer_mapping`; needed because some evosax algorithms
        read the model dimensionality from ``task`` when computing
        their default popsize.

    Returns
    -------
    int
        Resolved popsize.
    """
    update_class = optimizer_mapping(normalize_key(opt_step.optimizer))(
        task=task,
        opt_hash=opt_step.hash,
        bounded=(None, None),
        stop_fn=opt_step.stopping_fn,
        **dict(opt_step.hyperparameters),
    )
    return int(update_class.popsize)


def optstep_to_subopt(
    opt_step: OptimizationStep,
    task: TaskLike,
) -> SubOpt:
    """Convert one :class:`OptimizationStep` to a :class:`SubOpt`.

    Dispatch is by ``opt_step.optimizer`` against
    :data:`~l2co_optimizers._src.optax_implementations.
    normalized_optax_normal`, :data:`...normalized_optax_from_state`,
    :data:`l2co_native_optax` (l2co's own optax-style transforms, e.g.
    ``'lbfgs'``), :data:`~l2co_optimizers._src.evosax_implementations.
    normalized_evosax`, and :data:`l2co_native_evosax` (l2co's own
    evosax-style algorithms, e.g. ``'shade'``). Population size is
    resolved by :func:`resolve_popsize`.

    Note that this path has no ``bounded`` channel: algorithms whose
    internals use box bounds (SHADE's midpoint boundary repair and
    turning-based mutation) receive them as ordinary ``x_min`` /
    ``x_max`` hyperparameters instead; without them SHADE runs
    unbounded (its repair degrades to a no-op).

    Nor does it have a per-evaluation-key channel: the ``value_fn``
    reaching a line search here is the caller's plain ``params ->
    scalar`` callable, with no key to split. ``'lbfgs'`` therefore
    resolves to :func:`~l2co_optimizers._src.lbfgs.stock_lbfgs` on
    *every* task, including stochastic ones — whereas the registry
    factory :func:`~l2co_optimizers._src.lbfgs.lbfgs_update` switches
    to :func:`~l2co_optimizers._src.lbfgs.lbfgs_per_eval_key` when
    ``task.pass_rng`` is set. On a stochastic task the noise contract
    of a wrapper-driven L-BFGS is whatever the caller's ``value_fn``
    implements, not the one-realization-per-evaluation contract the
    registry path guarantees — see
    ``l2co ADR 0008-lbfgs-on-the-subopt-path-takes-the-stock-
    linesearch.md``.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.optimizer`` is the name (normalised
        by lowercase/alphanumeric, matching ``l2co``'s registry),
        and ``opt_step.hyperparameters`` is forwarded to the
        underlying optax/evosax constructor (after stripping
        ``popsize`` itself and any :data:`CONSTRUCTOR_HYPERPARAMETERS`
        entries, which go to the algorithm constructor rather than
        its ``Params``).
    task : TaskLike
        Task containing the model whose trainable arrays size the
        evosax ``solution=`` template, and which is forwarded to
        :func:`resolve_popsize` for popsize resolution.

    Returns
    -------
    SubOpt
        Adapter ready to slot into a wrapper factory.

    Raises
    ------
    ValueError
        If ``opt_step.optimizer`` is not found in either registry.
    """
    name = normalize_key(opt_step.optimizer)
    hyperparams = dict(opt_step.hyperparameters)
    popsize = resolve_popsize(opt_step, task)

    # ``popsize`` has now been consumed by ``resolve_popsize``;
    # strip it before forwarding ``hyperparams`` to the optax /
    # evosax constructors below, neither of which accepts it as a
    # kwarg.
    hyperparams.pop("popsize", None)

    # --- Optax / gradient-based ----------------------------------------
    if (
        name in normalized_optax_normal
        or name in normalized_optax_from_state
        or name in l2co_native_optax
    ):
        if name in normalized_optax_normal:
            optax_cls = normalized_optax_normal[name]
        elif name in normalized_optax_from_state:
            optax_cls = normalized_optax_from_state[name]
        else:
            optax_cls = l2co_native_optax[name]
        transform = optax_cls(**hyperparams)
        bound_fn = LINESEARCH_FEVAL_BOUND.get(name)
        # Same builder the ``UpdateClass`` path calls, from the same
        # hyperparameters, so a deployed policy reads and writes scale
        # exactly as the training environment did.
        read_fn, write_fn = build_transfer_fns(
            name, FAMILY_GRADIENT, hyperparams
        )
        return grad_sub_optimizer(
            transform,
            name_hash=opt_step.hash,
            opt_state_handshake=handshake_policy_for(opt_step).opt_state,
            population_handshake=handshake_policy_for(opt_step).population,
            max_fevals_per_step=(
                None if bound_fn is None else bound_fn(hyperparams)
            ),
            family=FAMILY_GRADIENT,
            transfer_read_fn=read_fn,
            transfer_write_fn=write_fn,
        )

    # --- Evosax / population-based -------------------------------------
    if name in normalized_evosax or name in l2co_native_evosax:
        algo_cls = normalized_evosax.get(name) or l2co_native_evosax[name]
        # Shape-affecting hyperparameters go to the constructor, not to
        # ``default_params.replace`` (e.g. SHADE's memory/archive size).
        constructor_kwargs = {
            kwarg: hyperparams.pop(kwarg)
            for kwarg in CONSTRUCTOR_HYPERPARAMETERS.get(name, ())
            if kwarg in hyperparams
        }
        # evosax wants ``solution`` to infer per-candidate shape;
        # ``task.model``'s trainable arrays are the standard template
        # (matching ``RunState.init``).
        solution_template = eqx.filter(task.model, eqx.is_inexact_array)
        algo = algo_cls(
            population_size=popsize,
            solution=solution_template,
            **constructor_kwargs,
        )
        es_params = (
            algo.default_params.replace(**hyperparams)
            if hyperparams
            else algo.default_params
        )
        is_distribution_based = issubclass(
            algo_cls, DistributionBasedAlgorithm
        )
        is_population_based = issubclass(algo_cls, PopulationBasedAlgorithm)
        if not (is_distribution_based or is_population_based):
            raise ValueError(
                f"evosax algorithm {algo_cls.__name__} is not registered "
                f"as distribution-based or population-based; cannot "
                f"build a SubOpt automatically."
            )

        def init_fn(params: PyTree, key: PRNGKeyArray) -> PyTree:
            """Initialise the ES state around ``params``.

            Dispatches on the algorithm family (distribution-based
            vs population-based):

            * **Distribution-based** (CMA-ES, SepCMA-ES, …) — init
              takes the single-instance ``params`` as the mean of
              the initial distribution.
            * **Population-based** (PSO, DE, …) — init takes an
              initial population (broadcast of ``params``) plus
              placeholder fitnesses; the ES updates them on the
              first ``tell``.

            Parameters
            ----------
            params : PyTree
                Single-instance parameter pytree.
            key : PRNGKeyArray
                ES initialisation key.

            Returns
            -------
            PyTree
                evosax algorithm state.
            """
            if is_distribution_based:
                return algo.init(key, params, es_params)
            pop0 = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (popsize, *x.shape)),
                params,
            )
            # Placeholder fitness is +inf, not zero: greedy-selection
            # algorithms (DE, SHADE) only replace a parent when the
            # trial's fitness beats the stored one, so a zero
            # placeholder would freeze the population until a trial
            # reached a negative loss.
            fit0 = jnp.full((popsize,), jnp.inf, dtype=float)
            return algo.init(
                key=key,
                population=pop0,
                fitness=fit0,
                params=es_params,
            )

        def ask_fn(
            params: PyTree,
            sub_state: PyTree,
            key: PRNGKeyArray,
        ) -> tuple[PyTree, PyTree]:
            """Ask the ES for a fresh population.

            Parameters
            ----------
            params : PyTree
                Current parameters (unused — the ES proposes its
                own center).
            sub_state : PyTree
                Current ES state.
            key : PRNGKeyArray
                Sampling key.

            Returns
            -------
            tuple[PyTree, PyTree]
                ``(population, new_sub_state)`` with ``population``
                of shape ``(popsize, *params_shape)``.
            """
            del params
            return algo.ask(key=key, state=sub_state, params=es_params)

        def tell_fn(
            sub_state: PyTree,
            population: PyTree,
            fitnesses: Float[Array, " popsize"],
            key: PRNGKeyArray,
        ) -> PyTree:
            """Tell the ES the previous round's fitnesses.

            Parameters
            ----------
            sub_state : PyTree
                Current ES state.
            population : PyTree
                The population that was evaluated, shape
                ``(popsize, *params_shape)``.
            fitnesses : Float[Array, " popsize"]
                Per-candidate losses (to be minimised).
            key : PRNGKeyArray
                Sampling key.

            Returns
            -------
            PyTree
                Updated ES state.
            """
            new_state, _ = algo.tell(
                key=key,
                population=population,
                fitness=fitnesses,
                state=sub_state,
                params=es_params,
            )
            return new_state

        # ``is_distribution_based`` was already resolved above from the
        # evosax class hierarchy; reuse it rather than re-deriving the
        # family from registry membership, which ``FEEDBACK_evosax.md``
        # records as fragile.
        family = (
            FAMILY_DISTRIBUTION if is_distribution_based else FAMILY_POPULATION
        )
        spec = transfer_spec_for(name, family)
        read_fn, write_fn = build_transfer_fns(
            name,
            family,
            hyperparams,
            ravel_fn=(algo._ravel_solution if is_distribution_based else None),
        )
        return pop_sub_optimizer(
            init_fn=init_fn,
            ask_fn=ask_fn,
            tell_fn=tell_fn,
            popsize=popsize,
            name_hash=opt_step.hash,
            opt_state_handshake=handshake_policy_for(opt_step).opt_state,
            population_handshake=handshake_policy_for(opt_step).population,
            family=family,
            transfer_read_fn=read_fn,
            transfer_write_fn=write_fn,
            own_ask=spec.own_ask,
        )

    raise ValueError(
        f"Optimizer {opt_step.optimizer!r} is not resolvable to a "
        f"SubOpt: it is absent from normalized_optax_normal, "
        f"normalized_optax_from_state, l2co_native_optax, "
        f"normalized_evosax and l2co_native_evosax. Note these are "
        f"narrower than l2co's optimizer registry — an optimizer that "
        f"``optimizer_mapping`` resolves may still be missing here. "
        f"Add it to the matching registry, or pass a hand-built "
        f"SubOpt via grad_sub_optimizer/pop_sub_optimizer."
    )
