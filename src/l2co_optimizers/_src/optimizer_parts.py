"""A bare optimizer unpacked into the pieces a switching loop drives.

The ``UpdateClass`` factories reduce an optimizer to one fused
``step_fn``. A loop that *switches* between optimizers -- l2co's
strategy wrapper, rl2co's deployment wrappers -- needs the pieces
instead: the optax ``GradientTransformation`` itself, or an evosax-style
``(init, ask, tell)`` triple, so it can interleave its own evaluation
and handshake between them. :func:`optimizer_parts` resolves an
:class:`~l2co_optimizers._src.core.optimizer_schedule.OptimizationStep`
to those pieces, as a :class:`GradientParts` or a
:class:`PopulationParts`.

Everything here is a property of the optimizer: which library object a
name resolves to, how its hyperparameters split between constructor and
``Params``, what one step costs in evaluations, and its state-transfer
ports (:mod:`~l2co_optimizers._src.core.state_transfer`). What is *not*
here is any decision about switching -- which handshake the optimizer
gets, how a menu dispatches across optimizers. That is l2co's switching
layer, which wraps these parts into its ``SubOpt`` adapter (l2co ADR
0019; the adapter itself was promoted from rl2co in l2co ADR 0003).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.optimizer_schedule import OptimizationStep
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    build_transfer_fns,
    transfer_spec_for,
)
from l2co_optimizers._src.core.typing import LossFunction
from l2co_optimizers._src.core.utils import normalize_key
from l2co_optimizers._src.evosax_implementations import normalized_evosax
from l2co_optimizers._src.ipopt import IPOPT_OPTIMIZERS
from l2co_optimizers._src.lbfgs import (
    DEFAULT_MAX_LINESEARCH_STEPS,
    stock_lbfgs,
)
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)
from l2co_optimizers._src.optimistix_implementations import (
    OPTIMISTIX_OPTIMIZERS,
)
from l2co_optimizers._src.rbf_trust_region import RBFTrustRegion
from l2co_optimizers._src.scipy_implementations import SCIPY_OPTIMIZERS
from l2co_optimizers._src.shade import SHADE
from l2co_optimizers._src.turbo import TuRBO

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "CONSTRUCTOR_HYPERPARAMETERS",
    "LINESEARCH_FEVAL_BOUND",
    "GradientParts",
    "PopulationParts",
    "l2co_native_evosax",
    "l2co_native_optax",
    "optimizer_parts",
    "resolve_popsize",
]


# l2co-native evosax-style algorithms: absent from evosax's own
# registries but subclassing its base API, so the generic evosax branch
# of ``optimizer_parts`` drives them unchanged.
# Keys are the normalized names (``normalize_key``), which is what
# ``optimizer_parts`` dispatches on -- an underscored name registered
# raw resolves directly but not as parts.
l2co_native_evosax: dict[str, type] = {
    "shade": SHADE,
    "rbftrustregion": RBFTrustRegion,
    "turbo": TuRBO,
}

# l2co-native optax-style transforms: assembled here rather than named
# on the ``optax`` module, so they are absent from the
# ``normalized_optax_*`` registries but the generic gradient branch of
# ``optimizer_parts`` drives them unchanged. ``"lbfgs"`` lives here
# because it moved out of ``opt_names_from_state`` into its own module
# (``l2co_optimizers._src.lbfgs``) to gain per-eval-key semantics; the
# parts path takes the stock variant, which is the only one it can
# drive (see :func:`optimizer_parts`).
l2co_native_optax: dict[str, Callable[..., optax.GradientTransformation]] = {
    "lbfgs": stock_lbfgs,
}

# Gradient transforms that spend more than one evaluation per step, keyed
# as in the registry, mapping their hyperparameters to the worst case.
# A line search evaluates the objective once per trial, so its cost is
# unrelated to its population of one -- consumers that size a budget or
# normalise a per-evaluation quantity must use the bound rather than
# ``popsize`` (see ``l2co ADR 0010``). Mirrors what
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


#                                                                     Parts
# =============================================================================


class GradientParts(eqx.Module):
    """A gradient-based optimizer, unpacked.

    All fields are static: the parts contribute no array leaves to a
    surrounding pytree.

    Attributes
    ----------
    transform : optax.GradientTransformation
        The optax transform, hyperparameters closed over.
    name_hash : int
        Hash of the source :class:`OptimizationStep`.
    max_fevals_per_step : int | None
        Worst-case evaluations one step can spend, for line-search
        transforms (:data:`LINESEARCH_FEVAL_BOUND`); ``None`` when a step
        costs exactly one. The actual count of a given step comes from
        :func:`~l2co_optimizers._src.optax_implementations.step_fevals`.
    family : str
        :data:`~l2co_optimizers._src.core.state_transfer.FAMILY_GRADIENT`.
    transfer_read_fn : Callable
        ``transfer_read_fn(opt_state) -> (sigma, conf)``; built by
        :func:`~l2co_optimizers._src.core.state_transfer.build_transfer_fns`
        from the same hyperparameters, exactly as the ``UpdateClass``
        factory does.
    transfer_write_fn : Callable
        ``transfer_write_fn(bundle, opt_state) -> opt_state``.
    popsize : int
        Always ``1``.
    """

    transform: optax.GradientTransformation = eqx.field(static=True)
    name_hash: int = eqx.field(static=True)
    max_fevals_per_step: int | None = eqx.field(static=True)
    family: str = eqx.field(static=True)
    transfer_read_fn: Callable = eqx.field(static=True)
    transfer_write_fn: Callable = eqx.field(static=True)
    popsize: int = eqx.field(static=True, default=1)


class PopulationParts(eqx.Module):
    """A population-based optimizer, unpacked into ``(init, ask, tell)``.

    All fields are static: the parts contribute no array leaves to a
    surrounding pytree.

    Attributes
    ----------
    init_fn : Callable
        ``init_fn(params, key) -> opt_state``.
    ask_fn : Callable
        ``ask_fn(params, opt_state, key) -> (population, opt_state)``,
        with ``population`` of shape ``(popsize, *params_shape)``.
    tell_fn : Callable
        ``tell_fn(opt_state, population, fitnesses, key) -> opt_state``.
    popsize : int
        Candidates per ``ask``.
    name_hash : int
        Hash of the source :class:`OptimizationStep`.
    family : str
        :data:`~l2co_optimizers._src.core.state_transfer.FAMILY_DISTRIBUTION`
        or :data:`~l2co_optimizers._src.core.state_transfer.FAMILY_POPULATION`,
        from the evosax class hierarchy.
    transfer_read_fn : Callable
        ``transfer_read_fn(opt_state) -> (sigma, conf)``.
    transfer_write_fn : Callable
        ``transfer_write_fn(bundle, opt_state) -> opt_state``.
    own_ask : bool
        Whether the optimizer draws its own post-switch population
        (:attr:`~l2co_optimizers._src.core.state_transfer.TransferSpec.own_ask`,
        ``l2co ADR 0014``).
    """

    init_fn: Callable = eqx.field(static=True)
    ask_fn: Callable = eqx.field(static=True)
    tell_fn: Callable = eqx.field(static=True)
    popsize: int = eqx.field(static=True)
    name_hash: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    transfer_read_fn: Callable = eqx.field(static=True)
    transfer_write_fn: Callable = eqx.field(static=True)
    own_ask: bool = eqx.field(static=True, default=False)


#                                                                  Resolution
# =============================================================================


def resolve_popsize(
    opt_step: OptimizationStep,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
) -> int:
    """Resolve the canonical popsize for ``opt_step`` via the registry.

    Builds an :class:`~l2co_optimizers._src.core.update_class.UpdateClass`
    purely to read its ``popsize`` attribute — ``1`` for optax-style
    optimizers, an explicit value or ``int(4 + 3 * log(d))`` for
    evosax — so the parts' population sizing matches the call site
    in :meth:`~l2co_optimizers.RunState.init`. The ``UpdateClass``
    instance itself is discarded.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.hyperparameters`` (including any
        explicit ``popsize`` kwarg) is forwarded to the factory.
    model : PyTree
        Forwarded to the factory; its parameter count drives the
        default popsize of the population-based optimizers.
    loss_fn : LossFunction
        Forwarded to the factory, which needs one to build.
    pass_rng : bool
        Forwarded to the factory, which needs one to build.

    Returns
    -------
    int
        Resolved popsize.
    """
    update_class = optimizer_mapping(normalize_key(opt_step.optimizer))(
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_step.hash,
        bounded=(None, None),
        stop_fn=opt_step.stopping_fn,
        **dict(opt_step.hyperparameters),
    )
    return int(update_class.popsize)


def optimizer_parts(
    opt_step: OptimizationStep,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
) -> GradientParts | PopulationParts:
    """Unpack one :class:`OptimizationStep` into the pieces a loop drives.

    Dispatch is by ``opt_step.optimizer`` against
    :data:`~l2co_optimizers._src.optax_implementations.
    normalized_optax_normal`, :data:`...normalized_optax_from_state`,
    :data:`l2co_native_optax` (e.g. ``'lbfgs'``), :data:`~l2co_optimizers.
    _src.evosax_implementations.normalized_evosax`, and
    :data:`l2co_native_evosax` (e.g. ``'shade'``). Population size is
    resolved by :func:`resolve_popsize`, so it always equals the
    ``UpdateClass`` factory's.

    Note that this path has no ``bounded`` channel: algorithms whose
    internals use box bounds (SHADE's midpoint boundary repair and
    turning-based mutation) receive them as ordinary ``x_min`` /
    ``x_max`` hyperparameters instead; without them SHADE runs
    unbounded (its repair degrades to a no-op).

    Nor does it have a per-evaluation-key channel: the ``value_fn``
    reaching a line search here is the caller's plain ``params ->
    scalar`` callable, with no key to split. ``'lbfgs'`` therefore
    resolves to :func:`~l2co_optimizers._src.lbfgs.stock_lbfgs` on
    *every* problem, including stochastic ones — whereas the registry
    factory :func:`~l2co_optimizers._src.lbfgs.lbfgs_update` switches
    to :func:`~l2co_optimizers._src.lbfgs.lbfgs_per_eval_key` when
    ``pass_rng`` is set. On a stochastic task the noise contract
    of a wrapper-driven L-BFGS is whatever the caller's ``value_fn``
    implements, not the one-realization-per-evaluation contract the
    registry path guarantees — see
    ``l2co ADR 0008-lbfgs-on-the-subopt-path-takes-the-stock-
    linesearch.md``.

    Parameters
    ----------
    opt_step : OptimizationStep
        Source spec; ``opt_step.optimizer`` is the name (normalised
        by lowercase/alphanumeric, matching the registry), and
        ``opt_step.hyperparameters`` is forwarded to the underlying
        optax/evosax constructor (after stripping ``popsize`` itself
        and any :data:`CONSTRUCTOR_HYPERPARAMETERS` entries, which go
        to the algorithm constructor rather than its ``Params``).
    model : PyTree
        Model whose trainable arrays size the evosax ``solution=``
        template; forwarded with ``loss_fn`` and ``pass_rng`` to
        :func:`resolve_popsize`.
    loss_fn : LossFunction
        Forwarded to :func:`resolve_popsize`.
    pass_rng : bool
        Forwarded to :func:`resolve_popsize`.

    Returns
    -------
    GradientParts | PopulationParts
        The unpacked optimizer, with no handshake decision attached.

    Raises
    ------
    ValueError
        If ``opt_step.optimizer`` is not found in any of the registries,
        including when it names one of the optimistix minimisers (ADR
        0001), the scipy minimisers (ADR 0002) or IPOPT (ADR 0003),
        which have no parts representation.
    """
    name = normalize_key(opt_step.optimizer)
    hyperparams = dict(opt_step.hyperparameters)
    popsize = resolve_popsize(
        opt_step, model=model, loss_fn=loss_fn, pass_rng=pass_rng
    )

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
        return GradientParts(
            transform=transform,
            name_hash=int(opt_step.hash),
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
        # ``model``'s trainable arrays are the standard template
        # (matching ``RunState.init``).
        solution_template = eqx.filter(model, eqx.is_inexact_array)
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
                f"unpack it automatically."
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
            opt_state: PyTree,
            key: PRNGKeyArray,
        ) -> tuple[PyTree, PyTree]:
            """Ask the ES for a fresh population.

            Parameters
            ----------
            params : PyTree
                Current parameters (unused — the ES proposes its
                own center).
            opt_state : PyTree
                Current ES state.
            key : PRNGKeyArray
                Sampling key.

            Returns
            -------
            tuple[PyTree, PyTree]
                ``(population, new_opt_state)`` with ``population``
                of shape ``(popsize, *params_shape)``.
            """
            del params
            return algo.ask(key=key, state=opt_state, params=es_params)

        def tell_fn(
            opt_state: PyTree,
            population: PyTree,
            fitnesses: Float[Array, " popsize"],
            key: PRNGKeyArray,
        ) -> PyTree:
            """Tell the ES the previous round's fitnesses.

            Parameters
            ----------
            opt_state : PyTree
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
                state=opt_state,
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
        return PopulationParts(
            init_fn=init_fn,
            ask_fn=ask_fn,
            tell_fn=tell_fn,
            popsize=int(popsize),
            name_hash=int(opt_step.hash),
            family=family,
            transfer_read_fn=read_fn,
            transfer_write_fn=write_fn,
            own_ask=spec.own_ask,
        )

    if name in OPTIMISTIX_OPTIMIZERS:
        raise ValueError(
            f"Optimizer {opt_step.optimizer!r} is an optimistix minimiser, "
            f"which cannot be unpacked into parts: it evaluates the "
            f"objective itself, so it is neither an optax transform "
            f"(GradientParts) nor an ask/tell population (PopulationParts). "
            f"It runs only as a plain registry entry, not inside a "
            f"switching menu (ADR 0001)."
        )

    if name in SCIPY_OPTIMIZERS:
        raise ValueError(
            f"Optimizer {opt_step.optimizer!r} is a scipy minimiser, which "
            f"cannot be unpacked into parts: scipy owns the optimization "
            f"loop and runs the whole budget inside one host callback, so "
            f"there is no step to interleave with a switch. It runs only "
            f"as a plain registry entry, not inside a switching menu (ADR "
            f"0002)."
        )

    if name in IPOPT_OPTIMIZERS:
        raise ValueError(
            f"Optimizer {opt_step.optimizer!r} is IPOPT, which cannot be "
            f"unpacked into parts: IPOPT owns the optimization loop and "
            f"runs the whole budget inside one host callback, so there is "
            f"no step to interleave with a switch. It runs only as a plain "
            f"registry entry, not inside a switching menu (ADR 0003)."
        )

    raise ValueError(
        f"Optimizer {opt_step.optimizer!r} cannot be unpacked into "
        f"parts: it is absent from normalized_optax_normal, "
        f"normalized_optax_from_state, l2co_native_optax, "
        f"normalized_evosax and l2co_native_evosax. Note these are "
        f"narrower than the optimizer registry — an optimizer that "
        f"``optimizer_mapping`` resolves may still be missing here. "
        f"Add it to the matching registry, or hand-build the switching "
        f"adapter in l2co."
    )
