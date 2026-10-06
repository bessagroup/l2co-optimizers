"""
SHADE — Success-History based Adaptive Differential Evolution.

Implements Tanabe & Fukunaga (2013) as specified in Sun et al. (2020),
"Success history-based adaptive differential evolution using
turning-based mutation" (Mathematics 8(9), 1565), including the paper's
optional turning-based mutation (``use_turning``). The algorithm is an
EvoSax :class:`PopulationBasedAlgorithm`, so the generic
``evosax_population_based_fn`` adapter drives it: the framework
evaluates fitness and calls ``tell`` (selection, archive and
success-history update) followed by ``ask`` (mutation + crossover).

Deviations from the paper's per-individual pseudo-code (all standard
for vectorized SHADE implementations):

1. **Synchronous generations** — all ``NP`` trial vectors are built
   from the generation-``G`` parents and archive; a parent defeated
   mid-generation cannot enter the archive for a later individual of
   the same generation.
2. **Archive overflow** — defeated parents are inserted in one batch
   and the archive is uniformly down-sampled to capacity, instead of
   per-insertion random deletion.
3. **Scale-factor sampling** — ``F`` is drawn from the Cauchy
   distribution truncated to ``(0, inf)`` via inverse-CDF sampling
   (then clipped at 1), which is statistically identical to the
   paper's resample-until-positive loop.
4. **NaN robustness** — fitness values are sanitized ``NaN -> inf``;
   replacement uses ``<=`` while success (archive insertion and
   memory update) requires a strict ``<``, so the weighted means never
   see NaN or zero total improvement.
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
import jax.random as jr
from evosax.algorithms.population_based.base import (
    Params as BasePopulationParams,
)
from evosax.algorithms.population_based.base import (
    PopulationBasedAlgorithm,
    metrics_fn,
)
from evosax.algorithms.population_based.base import (
    State as BasePopulationState,
)
from evosax.core.fitness_shaping import identity_fitness_shaping_fn
from evosax.types import Fitness, Population, Solution

# TODO: drop the flax dependency -- it is used only for ``flax.struct``
# (shade.py, turbo.py, rbf_trust_region.py); equinox modules
# or plain dataclasses would do.
from flax import struct
from jax.tree_util import Partial
from jaxtyping import PyTree

# Local
from l2co_optimizers._src.core.popsize import resolve_popsize, shade_popsize
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_POPULATION,
    build_transfer_fns,
)
from l2co_optimizers._src.core.typing import (
    LossFunction,
    PopSize,
    StopFunction,
)
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.evosax_implementations import (
    evosax_population_based_fn,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _truncated_cauchy(
    key: jax.Array, location: jax.Array, scale: float = 0.1
) -> jax.Array:
    """Sample ``Cauchy(location, scale)`` truncated to ``(0, inf)``.

    Inverse-CDF sampling; statistically identical to the paper's
    resample-until-positive loop (Eq. 7) before the clip at 1 that the
    caller applies. The result is floored at the smallest positive
    normal float so it is strictly positive.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    location : jax.Array
        Cauchy location parameter(s); output matches its shape.
    scale : float, optional
        Cauchy scale parameter.

    Returns
    -------
    jax.Array
        Strictly positive samples, same shape as ``location``.
    """
    cdf_at_zero = jnp.arctan(-location / scale) / jnp.pi + 0.5
    u = cdf_at_zero + (1.0 - cdf_at_zero) * jr.uniform(
        key, jnp.shape(location)
    )
    sample = location + scale * jnp.tan(jnp.pi * (u - 0.5))
    return jnp.maximum(sample, jnp.finfo(sample.dtype).tiny)


def _donor_indices(
    key_r1: jax.Array,
    key_r2: jax.Array,
    popsize: int,
    archive_count: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample the ``r1`` / ``r2`` donor indices for every individual.

    ``r1`` is drawn uniformly from the population excluding the
    individual itself; ``r2`` is drawn uniformly from the union of the
    population and the valid archive rows, excluding both the
    individual and ``r1``. Exclusions are exact (index-shift trick, no
    rejection loop).

    Parameters
    ----------
    key_r1, key_r2 : jax.Array
        PRNG keys.
    popsize : int
        Population size ``NP``.
    archive_count : jax.Array
        Number of valid archive rows (scalar int).

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(r1, r2)`` of shape ``(popsize,)``; ``r1`` indexes the
        population, ``r2`` indexes the union (values at or past
        ``popsize`` refer to archive row ``r2 - popsize``).
    """
    member_ids = jnp.arange(popsize)

    r1 = jr.randint(key_r1, (popsize,), 0, popsize - 1)
    r1 = r1 + (r1 >= member_ids)

    union_size = popsize + archive_count
    low = jnp.minimum(member_ids, r1)
    high = jnp.maximum(member_ids, r1)
    r2 = jr.randint(key_r2, (popsize,), 0, union_size - 2)
    r2 = r2 + (r2 >= low)
    r2 = r2 + (r2 >= high)
    return r1, r2


def _ring_radii(
    span: jax.Array,
    num_dims: int,
    fevals: jax.Array,
    max_fevals: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Inner and current outer radius of the turning ring (Eqs. 18-20).

    ``OR_init = sqrt(D) * span / 2`` (the radius of the search domain),
    ``IR = sqrt(D) * span / 40``; the outer radius shrinks linearly
    from ``OR_init`` to ``IR`` as ``fevals`` approaches ``max_fevals``
    (and stays at ``IR`` beyond it).

    Parameters
    ----------
    span : jax.Array
        Per-dimension search range ``x_max - x_min``.
    num_dims : int
        Problem dimensionality ``D``.
    fevals : jax.Array
        Fitness evaluations spent so far ``FES``.
    max_fevals : jax.Array
        Total evaluation budget ``MaxFES``.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(inner, outer)`` radii.
    """
    sqrt_dims = jnp.sqrt(jnp.asarray(num_dims, dtype=float))
    outer_init = sqrt_dims * span / 2.0
    inner = sqrt_dims * span / 40.0
    progress = jnp.minimum(fevals / max_fevals, 1.0)
    outer = outer_init + (inner - outer_init) * progress
    return inner, outer


@struct.dataclass
class State(BasePopulationState):
    """SHADE state.

    Attributes
    ----------
    archive : jax.Array
        External archive of defeated parents, shape
        ``(archive_size, num_dims)``; rows at or past ``archive_count``
        are NaN padding.
    archive_count : jax.Array
        Number of valid rows in ``archive`` (scalar int).
    memory_f : jax.Array
        Success-history memory ``M_F``, shape ``(memory_size,)``.
    memory_cr : jax.Array
        Success-history memory ``M_CR``, shape ``(memory_size,)``.
    memory_index : jax.Array
        Next memory cell ``k`` to overwrite (scalar int).
    scale_factors : jax.Array
        Per-individual ``F_i`` used to build the pending trial
        population, shape ``(population_size,)``.
    crossover_rates : jax.Array
        Per-individual ``CR_i`` used to build the pending trial
        population, shape ``(population_size,)``.
    """

    archive: jax.Array
    archive_count: jax.Array
    memory_f: jax.Array
    memory_cr: jax.Array
    memory_index: jax.Array
    scale_factors: jax.Array
    crossover_rates: jax.Array


@struct.dataclass
class Params(BasePopulationParams):
    """SHADE parameters (value-like; shapes live on the class).

    Attributes
    ----------
    p_max : float
        Upper end of the greediness range for current-to-pbest/1;
        ``p_i ~ U[2 / NP, p_max]`` (Eqs. 5-6).
    memory_init_f : float
        Initial fill value of ``M_F`` (Eq. 3).
    memory_init_cr : float
        Initial fill value of ``M_CR`` (Eq. 3).
    x_min : float
        Lower box bound; ``-inf`` disables midpoint boundary repair.
    x_max : float
        Upper box bound; ``+inf`` disables midpoint boundary repair.
    use_turning : bool
        Enable turning-based mutation (Sun et al. 2020, Operation 1).
        Requires finite bounds and a finite ``max_fevals``.
    max_fevals : float
        Total fitness-evaluation budget ``MaxFES`` driving the
        shrinking outer radius (Eq. 20). Only used when
        ``use_turning`` is enabled.
    """

    p_max: float = 0.2
    memory_init_f: float = 0.5
    memory_init_cr: float = 0.5
    x_min: float = -jnp.inf
    x_max: float = jnp.inf
    use_turning: bool = False
    max_fevals: float = jnp.inf


class SHADE(PopulationBasedAlgorithm):
    """Success-History based Adaptive DE with optional turning mutation.

    Attributes
    ----------
    memory_size : int
        Size ``H`` of the success-history memories ``M_F`` / ``M_CR``.
        Defaults to the population size (paper setting ``H = NP``).
    archive_size : int
        Capacity ``|A|`` of the external archive of defeated parents.
        Defaults to the population size (paper setting ``|A| = NP``).
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        memory_size: int | None = None,
        archive_size: int | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable = metrics_fn,
    ):
        """Initialize SHADE.

        Parameters
        ----------
        population_size : int
            Population size ``NP``; must be at least 4 so that
            ``i``, ``r1``, ``r2`` and a pbest individual can be
            distinct.
        solution : Solution
            Solution PyTree template (ravelled internally).
        memory_size : int or None, optional
            Success-history memory size ``H``; ``None`` uses ``NP``.
        archive_size : int or None, optional
            Archive capacity ``|A|``; ``None`` uses ``NP``.
        fitness_shaping_fn : Callable, optional
            EvoSax fitness shaping hook.
        metrics_fn : Callable, optional
            EvoSax metrics hook.
        """
        assert population_size >= 4, "SHADE requires population_size >= 4."
        super().__init__(
            population_size, solution, fitness_shaping_fn, metrics_fn
        )
        self.memory_size = (
            memory_size if memory_size is not None else population_size
        )
        self.archive_size = (
            archive_size if archive_size is not None else population_size
        )
        assert self.memory_size >= 1, "SHADE requires memory_size >= 1."
        assert self.archive_size >= 1, "SHADE requires archive_size >= 1."

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full(
                (self.population_size, self.num_dims), jnp.nan
            ),
            fitness=jnp.full(self.population_size, jnp.inf),
            archive=jnp.full((self.archive_size, self.num_dims), jnp.nan),
            archive_count=jnp.array(0, dtype=int),
            memory_f=jnp.full(self.memory_size, params.memory_init_f),
            memory_cr=jnp.full(self.memory_size, params.memory_init_cr),
            memory_index=jnp.array(0, dtype=int),
            scale_factors=jnp.full(self.population_size, 0.5),
            crossover_rates=jnp.full(self.population_size, 0.5),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.inf,
            generation_counter=0,
        )

    def _ask(
        self,
        key: jax.Array,
        state: State,
        params: Params,
    ) -> tuple[Population, State]:
        popsize = self.population_size
        num_dims = self.num_dims
        fitness = jnp.where(jnp.isnan(state.fitness), jnp.inf, state.fitness)

        (
            key_mem,
            key_f,
            key_cr,
            key_p,
            key_pbest,
            key_r1,
            key_r2,
            key_jrand,
            key_cross,
            key_ndims,
            key_perm,
            key_rand,
        ) = jr.split(key, 12)

        # Control parameters from the success-history memories (Eqs. 7-8):
        # F_i ~ Cauchy(M_F[ri], 0.1) truncated to (0, inf), clipped at 1,
        # via inverse-CDF sampling; CR_i ~ N(M_CR[ri], 0.1) clipped to
        # [0, 1].
        mem_ids = jr.randint(key_mem, (popsize,), 0, self.memory_size)
        m_f = state.memory_f[mem_ids]
        m_cr = state.memory_cr[mem_ids]

        scale_factor = jnp.minimum(_truncated_cauchy(key_f, m_f), 1.0)
        crossover_rate = jnp.clip(
            m_cr + 0.1 * jr.normal(key_cr, (popsize,)), 0.0, 1.0
        )

        # pbest individual from the best NP * p_i members (Eqs. 4-6). The
        # lower end 2 / NP is clipped against p_max so tiny populations
        # degrade to a fixed p rather than an empty range.
        p_low = jnp.minimum(2.0 / popsize, params.p_max)
        p = jr.uniform(key_p, (popsize,), minval=p_low, maxval=params.p_max)
        n_best = jnp.clip(jnp.round(p * popsize), 2, popsize).astype(int)
        order = jnp.argsort(fitness)
        pbest_ids = order[jr.randint(key_pbest, (popsize,), 0, n_best)]
        x_pbest = state.population[pbest_ids]

        # Donors: r1 from P \ {i}, r2 from (P u A) \ {i, r1}, both exact.
        r1, r2 = _donor_indices(key_r1, key_r2, popsize, state.archive_count)
        x_r1 = state.population[r1]
        from_archive = r2 >= popsize
        x_r2 = jnp.where(
            from_archive[:, None],
            state.archive[jnp.clip(r2 - popsize, 0, self.archive_size - 1)],
            state.population[jnp.clip(r2, 0, popsize - 1)],
        )

        # current-to-pbest/1 differential vector (Eqs. 4 and 22).
        x = state.population
        diff = scale_factor[:, None] * (x_pbest - x) + scale_factor[
            :, None
        ] * (x_r1 - x_r2)

        # Turning-based mutation (Sun et al. 2020, Eqs. 18-21 and
        # Operation 1): inside the ring around the pbest individual the
        # differential vector is negated and a random subset of its
        # dimensions is replaced by uniform values in the search range.
        # With the default infinite bounds / budget the ring condition is
        # never met, so this reduces to canonical SHADE.
        span = params.x_max - params.x_min
        fevals = popsize * (state.generation_counter + 1)
        inner, outer = _ring_radii(span, num_dims, fevals, params.max_fevals)
        distance = jnp.linalg.norm(x_pbest - x, axis=1)
        in_ring = params.use_turning & (distance > inner) & (distance < outer)

        n_turn_dims = jr.randint(key_ndims, (popsize,), 1, num_dims + 1)
        dim_ranks = jnp.argsort(
            jnp.argsort(jr.uniform(key_perm, (popsize, num_dims)), axis=1),
            axis=1,
        )
        turn_mask = dim_ranks < n_turn_dims[:, None]
        random_component = params.x_min + span * jr.uniform(
            key_rand, (popsize, num_dims)
        )
        turned_diff = jnp.where(turn_mask, random_component, -diff)
        diff = jnp.where(in_ring[:, None], turned_diff, diff)

        # Mutant vector (Eq. 23) with midpoint boundary repair (Eq. 2);
        # a no-op for infinite bounds.
        mutant = x + diff
        mutant = jnp.where(
            mutant < params.x_min, (params.x_min + x) / 2.0, mutant
        )
        mutant = jnp.where(
            mutant > params.x_max, (params.x_max + x) / 2.0, mutant
        )

        # Binomial crossover with a guaranteed j_rand dimension.
        j_rand = jax.nn.one_hot(
            jr.randint(key_jrand, (popsize,), 0, num_dims),
            num_dims,
            dtype=bool,
        )
        cross = jr.uniform(key_cross, (popsize, num_dims))
        cross_mask = (cross < crossover_rate[:, None]) | j_rand
        trials = jnp.where(cross_mask, mutant, x)

        state = state.replace(
            scale_factors=scale_factor, crossover_rates=crossover_rate
        )
        return trials, state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        trial_fitness = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)
        parent_fitness = jnp.where(
            jnp.isnan(state.fitness), jnp.inf, state.fitness
        )

        # Greedy selection: replace on <=, but only a strict improvement
        # counts as a success (feeds archive and memory update).
        replace = trial_fitness <= parent_fitness
        success = trial_fitness < parent_fitness

        new_population = jnp.where(
            replace[:, None], population, state.population
        )
        new_fitness = jnp.where(replace, trial_fitness, parent_fitness)

        # Archive update: batch-insert the defeated parents, then keep a
        # uniformly random subset of the valid entries at capacity.
        pool = jnp.concatenate([state.archive, state.population], axis=0)
        valid = jnp.concatenate(
            [jnp.arange(self.archive_size) < state.archive_count, success]
        )
        priority = jnp.where(
            valid, jr.uniform(key, (pool.shape[0],)), -jnp.inf
        )
        keep = jnp.argsort(-priority)[: self.archive_size]
        new_archive = pool[keep]
        new_count = jnp.minimum(
            state.archive_count + jnp.sum(success), self.archive_size
        )

        # Success-history memory update (Eqs. 9-13): weighted arithmetic
        # mean for CR, weighted Lehmer mean for F, weights proportional
        # to the fitness improvement. Non-finite total improvement (a
        # trial beating an inf-sanitized parent) falls back to uniform
        # weights over the successes.
        improvement = jnp.where(success, parent_fitness - trial_fitness, 0.0)
        total = jnp.sum(improvement)
        n_success = jnp.sum(success)
        any_success = n_success > 0
        tiny = jnp.finfo(improvement.dtype).tiny
        uniform_weights = success / jnp.maximum(n_success, 1)
        weights = jnp.where(
            jnp.isfinite(total) & (total > 0),
            improvement / jnp.maximum(total, tiny),
            uniform_weights,
        )

        mean_cr = jnp.sum(weights * state.crossover_rates)
        lehmer_f = jnp.sum(weights * state.scale_factors**2) / jnp.maximum(
            jnp.sum(weights * state.scale_factors), tiny
        )

        k = state.memory_index
        new_memory_cr = jnp.where(
            any_success,
            state.memory_cr.at[k].set(mean_cr),
            state.memory_cr,
        )
        new_memory_f = jnp.where(
            any_success,
            state.memory_f.at[k].set(lehmer_f),
            state.memory_f,
        )
        new_memory_index = jnp.where(
            any_success, (k + 1) % self.memory_size, k
        )

        return state.replace(
            population=new_population,
            fitness=new_fitness,
            archive=new_archive,
            archive_count=new_count,
            memory_f=new_memory_f,
            memory_cr=new_memory_cr,
            memory_index=new_memory_index,
        )


def shade_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    popsize: PopSize = shade_popsize,
    bounded: tuple[float, float] | None = (None, None),
    stop_fn: StopFunction | None = None,
    memory_size: int | None = None,
    archive_size: int | None = None,
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for :class:`SHADE`.

    Splits the hyperparameters into shape-affecting constructor
    arguments (``memory_size``, ``archive_size``, ``popsize``) and
    value-like :class:`Params` overrides, and injects the run's box
    bounds into ``Params.x_min`` / ``Params.x_max`` (explicit
    hyperparameter values win over the injected bounds).

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
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int or Callable[[int], int], optional
        Population size; defaults to :func:`shade_popsize`
        (``max(10, 4 + 3 * log(d))`` so ``p in [2 / NP, 0.2]`` is
        well-defined).
    bounded : tuple[float, float] or None, optional
        Box bounds applied to parameters after each step and injected
        into the algorithm for boundary repair and turning-based
        mutation.
    stop_fn : StopFunction or None, optional
        Stopping function.
    memory_size : int or None, optional
        Success-history memory size ``H``; ``None`` uses the population
        size (paper setting).
    archive_size : int or None, optional
        Archive capacity ``|A|``; ``None`` uses the population size
        (paper setting).
    **hyperparameters
        Override fields on :class:`Params` (``p_max``,
        ``memory_init_f``, ``memory_init_cr``, ``x_min``, ``x_max``,
        ``use_turning``, ``max_fevals``).

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.

    Raises
    ------
    ValueError
        If ``use_turning`` is enabled without finite bounds or without
        a finite ``max_fevals``.
    """
    popsize = resolve_popsize(popsize, model)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    optimizer = SHADE(
        population_size=popsize,
        solution=params,
        memory_size=memory_size,
        archive_size=archive_size,
    )

    low, high = bounded if bounded is not None else (None, None)
    bound_overrides = {}
    if low is not None:
        bound_overrides["x_min"] = low
    if high is not None:
        bound_overrides["x_max"] = high
    es_params = optimizer.default_params.replace(
        **{**bound_overrides, **hyperparameters}
    )

    if es_params.use_turning:
        if not (
            jnp.isfinite(es_params.x_min) and jnp.isfinite(es_params.x_max)
        ):
            raise ValueError(
                "SHADE with use_turning=True requires finite box bounds: "
                "run with bounded=(low, high) or pass x_min / x_max "
                "hyperparameters."
            )
        if not jnp.isfinite(es_params.max_fevals):
            raise ValueError(
                "SHADE with use_turning=True requires the max_fevals "
                "hyperparameter (the run's total fitness-evaluation "
                "budget) for the shrinking outer radius."
            )

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

    # SHADE keeps its population, external archive and success-history
    # memories. Only scale crosses a switch: the memories are
    # self-adaptation of SHADE's own mutation operator and mean nothing
    # to another algorithm.
    read_fn, write_fn = build_transfer_fns(
        "shade", FAMILY_POPULATION, hyperparameters
    )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=popsize,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_POPULATION,
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
    )


shade_mapping = {"shade": Partial(shade_update)}
