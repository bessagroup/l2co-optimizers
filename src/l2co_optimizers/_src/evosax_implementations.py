"""
EvoSax ``UpdateClass`` factories and optimizer name registry.

The closure factories ``evosax_distribution_based_fn`` /
``evosax_population_based_fn`` build the ``(init_fn, step_fn)`` pairs
for an EvoSax algorithm, and ``evosax_ask_fn`` exposes its sampling step
for a handshake. The ``UpdateClass`` factories
``evosax_distribution_update`` / ``evosax_population_update`` wrap them
into an :class:`~l2co_optimizers._src.update_class.UpdateClass`. The
registry binds each EvoSax algorithm directly to the appropriate factory
via :class:`jax.tree_util.Partial`. Population size defaults to
:func:`l2co_optimizers._src.popsize.variable_popsize` on each factory,
except for the algorithms in :data:`_EVEN_POPSIZE_REQUIRED`, which are
bound to :func:`l2co_optimizers._src.popsize.variable_popsize_even`
because they require an even population size. The caller can override
either default by supplying ``popsize`` in the ``OptimizationStep``
hyperparameters.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

# Third-party
from evosax.algorithms import algorithms
from evosax.algorithms.base import Params
from evosax.algorithms.base import State as EvoSaxState
from evosax.algorithms.distribution_based import distribution_based_algorithms
from evosax.algorithms.distribution_based.base import (
    DistributionBasedAlgorithm,
)
from evosax.algorithms.population_based import population_based_algorithms
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax.tree_util import Partial
from jaxtyping import PRNGKeyArray, PyTree

from l2co_optimizers._src.loss import (
    vmapped_loss,
    vmapped_loss_with_rng,
)

# Local
from l2co_optimizers._src.opt_history import OptHistory
from l2co_optimizers._src.popsize import (
    resolve_popsize,
    variable_popsize,
    variable_popsize_even,
)
from l2co_optimizers._src.state_transfer import (
    FAMILY_DISTRIBUTION,
    FAMILY_POPULATION,
    build_transfer_fns,
    transfer_spec_for,
)
from l2co_optimizers._src.typing import (
    AskFunction,
    InitFunction,
    InputParameters,
    LossFunction,
    PopSize,
    StepFunction,
    StopFunction,
)
from l2co_optimizers._src.update_class import UpdateClass
from l2co_optimizers._src.utils import normalize_key

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#                                                             Closure factories
# =============================================================================


def evosax_ask_fn(
    optimizer: DistributionBasedAlgorithm | PopulationBasedAlgorithm,
    es_params: Params,
    bounded: tuple[float | None, float | None] | None,
) -> AskFunction:
    """Expose an EvoSax algorithm's own sampling step.

    ``step_fn`` bundles evaluate/tell/ask into one call with the evosax
    algorithm captured in its closure, so a caller holding only an
    :class:`UpdateClass` cannot ask for a population without also
    spending a generation. This exposes just the ``ask``, which is what
    a handshake needs: write location and scale into the state, then let
    the optimizer draw its own population, so whatever its next update
    reads matches the population that gets told back to it. For a
    distribution-based algorithm that is the draws it caches (``z``/``y``
    for CR-FM-NES, ``pert`` for the persistent-ES family); for a mutation
    GA it is the ``state.fitness`` baseline the writer just repaired.
    ``ask`` is declared on the shared base (``algorithms/base.py:106``),
    so one closure serves both families.

    Bounds are clipped exactly as ``step_fn`` clips them, so a
    handshake population obeys the same box as a stepped one.

    Parameters
    ----------
    optimizer : DistributionBasedAlgorithm or PopulationBasedAlgorithm
        Constructed EvoSax algorithm instance.
    es_params : Params
        Its resolved parameters.
    bounded : tuple[float | None, float | None] or None
        Box bounds, applied after ``ask``. ``None``, or ``None`` on one
        side, leaves that side unbounded.

    Returns
    -------
    AskFunction
        ``ask_fn(opt_state, key) -> (params, opt_state)``.
    """
    if bounded is None:
        bounded = (None, None)

    def ask_fn(
        opt_state: EvoSaxState, key: PRNGKeyArray
    ) -> tuple[PyTree, EvoSaxState]:
        """Draw one population from the optimizer's own distribution."""
        params, new_opt_state = optimizer.ask(
            key=key, state=opt_state, params=es_params
        )
        params = jax.tree.map(lambda p: jnp.clip(p, *bounded), params)
        return params, new_opt_state

    return ask_fn


def evosax_distribution_based_fn(
    static: PyTree,
    optimizer: DistributionBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None] | None,
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax distribution-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : DistributionBasedAlgorithm
        EvoSax distribution-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float | None, float | None] or None
        Parameter bounds for clipping. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
    opt_hash : int
        Hash for optimizer step tracking.
    pass_rng : bool
        Whether to pass RNG key to loss function.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """
    if bounded is None:
        bounded = (None, None)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray = None, **sample
    ) -> EvoSaxState:
        """Initialise the distribution-based ES state."""
        params = jax.tree.map(lambda x: x[0], params)
        return optimizer.init(key, params, es_params)

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one distribution-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )

        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            grads=grads,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(popsize, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


def evosax_population_based_fn(
    static: PyTree,
    optimizer: PopulationBasedAlgorithm,
    loss_fn: LossFunction,
    bounded: tuple[float | None, float | None] | None,
    popsize: int,
    es_params: Params,
    opt_hash: int,
    pass_rng: bool,
    **kwargs,
) -> tuple[InitFunction, StepFunction]:
    """
    Creates initialization and step functions for EvoSax population-based
    optimizers.

    Parameters
    ----------
    static : PyTree
        Static model parameters.
    optimizer : PopulationBasedAlgorithm
        EvoSax population-based optimizer instance.
    loss_fn : LossFunction
        Loss function to optimize.
    bounded : tuple[float | None, float | None] or None
        Parameter bounds for clipping. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    popsize : int
        Population size.
    es_params : Params
        Evolution strategy parameters.
    opt_hash : int
        Hash for optimizer step tracking.
    **kwargs
        Additional arguments.

    Returns
    -------
    tuple[InitFunction, StepFunction]
        Initialization and step functions for optimization.
    """
    if bounded is None:
        bounded = (None, None)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray, **sample
    ) -> EvoSaxState:
        """Initialise the population-based ES state."""
        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(eqx.combine(params, static), loss_fn, sample)

        return optimizer.init(
            key=key, population=params, fitness=loss, params=es_params
        )

    def step_fn(
        carry: tuple[InputParameters, EvoSaxState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, EvoSaxState, PRNGKeyArray], OptHistory]:
        """Perform one population-based ES step."""
        input_params, opt_state, key = carry

        if pass_rng:
            loss = vmapped_loss_with_rng(
                eqx.combine(input_params, static),
                loss_fn,
                sample,
                jr.split(key, popsize),
            )
        else:
            loss = vmapped_loss(
                eqx.combine(input_params, static), loss_fn, sample
            )

        new_opt_state, _ = optimizer.tell(
            key=key,
            population=input_params,
            fitness=loss,
            state=opt_state,
            params=es_params,
        )

        new_params, new_opt_state = optimizer.ask(
            key=key, state=new_opt_state, params=es_params
        )

        grads = jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), input_params)

        history = OptHistory(
            loss=loss,
            params=input_params,
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(popsize, dtype=int),
            grads=grads,
            iterations=jnp.array(1, dtype=int),
        )

        # Apply box constraints
        new_params = jax.tree.map(lambda p: jnp.clip(p, *bounded), new_params)

        new_key, _ = jr.split(key)
        return (new_params, new_opt_state, new_key), history

    return init_fn, step_fn


#                                                         UpdateClass factories
# =============================================================================


def evosax_distribution_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    optimizer: type[DistributionBasedAlgorithm],
    opt_hash: int,
    popsize: PopSize = variable_popsize,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    name: str = "",
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for an EvoSax distribution-based
    algorithm (CMA-ES, OpenAI-ES, ...).

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    optimizer : type[DistributionBasedAlgorithm]
        EvoSax algorithm class.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int or Callable[[int], int], optional
        Population size; either an integer or a callable taking the
        ``Task`` and returning an integer. Defaults to
        :func:`variable_popsize` (``int(4 + 3 * log(d))``).
    bounded : tuple[float | None, float | None] or None, optional
        Box bounds applied to parameters after each step. Defaults to
        ``(None, None)``; ``None``, or ``None`` on one side, leaves that
        side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.
    name : str, optional
        Registry name, used only to resolve this optimizer's
        state-transfer overrides
        (:data:`~l2co_optimizers._src.state_transfer.TRANSFER_OVERRIDES`).
        Declared explicitly rather than left to
        ``**hyperparameters`` so it is never forwarded to the
        underlying constructor. Defaults to ``""``, which resolves
        to the family default.
    **hyperparameters
        Override fields on ``optimizer.default_params``.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    popsize = resolve_popsize(popsize, model)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)
    es_params = optimizer.default_params.replace(**hyperparameters)

    init_fn, step_fn = evosax_distribution_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=es_params,
        opt_hash=opt_hash,
        pass_rng=pass_rng,
    )
    read_fn, write_fn = build_transfer_fns(
        name,
        FAMILY_DISTRIBUTION,
        hyperparameters,
        ravel_fn=optimizer._ravel_solution,
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_DISTRIBUTION,
        ask_fn=evosax_ask_fn(optimizer, es_params, bounded),
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
    )


def evosax_population_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    optimizer: type[PopulationBasedAlgorithm],
    opt_hash: int,
    popsize: PopSize = variable_popsize,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    name: str = "",
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for an EvoSax population-based
    algorithm (DE, PSO, ...).

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    optimizer : type[PopulationBasedAlgorithm]
        EvoSax algorithm class.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int or Callable[[int], int], optional
        Population size; either an integer or a callable taking the
        ``Task`` and returning an integer. Defaults to
        :func:`variable_popsize` (``int(4 + 3 * log(d))``).
    bounded : tuple[float | None, float | None] or None, optional
        Box bounds applied to parameters after each step. Defaults to
        ``(None, None)``; ``None``, or ``None`` on one side, leaves that
        side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.
    name : str, optional
        Registry name, used only to resolve this optimizer's
        state-transfer overrides
        (:data:`~l2co_optimizers._src.state_transfer.TRANSFER_OVERRIDES`).
        Declared explicitly rather than left to
        ``**hyperparameters`` so it is never forwarded to the
        underlying constructor. Defaults to ``""``, which resolves
        to the family default.
    **hyperparameters
        Override fields on ``optimizer.default_params``.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    popsize = resolve_popsize(popsize, model)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    optimizer = optimizer(population_size=popsize, solution=params)
    es_params = optimizer.default_params.replace(**hyperparameters)

    init_fn, step_fn = evosax_population_based_fn(
        static=static,
        optimizer=optimizer,
        loss_fn=loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=es_params,
        opt_hash=opt_hash,
        pass_rng=pass_rng,
    )
    spec = transfer_spec_for(name, FAMILY_POPULATION)
    read_fn, write_fn = build_transfer_fns(
        name, FAMILY_POPULATION, hyperparameters
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_POPULATION,
        # Only the mutation GAs opt in: their next generation is
        # scored against the baseline the writer just repaired, so
        # they have to draw it themselves. DE and the archive
        # methods must not -- their ``ask`` builds mutants from
        # difference vectors of the stored population, which the
        # repair makes identical, and identical rows have no
        # differences. See ``l2co ADR 0014``.
        ask_fn=(
            evosax_ask_fn(optimizer, es_params, bounded)
            if spec.own_ask
            else None
        ),
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
    )


#                                                                      Registry
# =============================================================================


normalized_evosax = {normalize_key(k): v for k, v in algorithms.items()}

# remove svcmaes and svopenes from the mapping as they require special
# treatment
_ = normalized_evosax.pop("svcmaes", None)
_ = normalized_evosax.pop("svopenes", None)
_ = normalized_evosax.pop("evotfes", None)
_ = normalized_evosax.pop("les", None)
_ = normalized_evosax.pop("lga", None)
# l2co ships its own random search under the same normalized key
# (``random_search.random_search_mapping``), which shadows evosax's.
_ = normalized_evosax.pop("randomsearch", None)


# EvoSax algorithms (normalized names) that require an even population
# size. The antithetic / mirrored-sampling strategies assert
# ``population_size % 2 == 0`` at construction, and ESMC silently returns
# ``popsize - 1`` candidates for an odd population size, which then
# mismatches the ``popsize``-shaped buffers downstream. These default to
# :func:`variable_popsize_even` instead of :func:`variable_popsize`.
_EVEN_POPSIZE_REQUIRED = frozenset(
    {
        "ars",
        "asebo",
        "crfmnes",
        "esmc",
        "guidedes",
        "noisereusees",
        "openes",
        "persistentes",
        "pgpe",
    }
)


def _factory_for(evosax_class: type):
    """Pick the ``UpdateClass`` factory matching an EvoSax algorithm."""
    if evosax_class in distribution_based_algorithms.values():
        return evosax_distribution_update
    if evosax_class in population_based_algorithms.values():
        return evosax_population_update
    raise ValueError(f"EvoSax class '{evosax_class.__name__}' not recognized.")


def _evosax_partial(name: str, cls: type) -> Partial:
    """Bind ``cls`` to its ``UpdateClass`` factory.

    Algorithms in :data:`_EVEN_POPSIZE_REQUIRED` get
    :func:`variable_popsize_even` as their default population size; the
    rest fall back to the factory's :func:`variable_popsize` default.
    """
    factory = _factory_for(cls)
    if name in _EVEN_POPSIZE_REQUIRED:
        return Partial(
            factory,
            optimizer=cls,
            popsize=variable_popsize_even,
            name=name,
        )
    return Partial(factory, optimizer=cls, name=name)


evosax_mapping = {
    name: _evosax_partial(name, cls) for name, cls in normalized_evosax.items()
}
