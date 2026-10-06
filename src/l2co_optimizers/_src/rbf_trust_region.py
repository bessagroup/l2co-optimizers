"""Radial-basis-function trust-region search (ORBIT / DYCORS family).

A model-based derivative-free method: it fits a cheap surrogate to the
points it has already evaluated, and spends its next evaluations where
the surrogate says the objective is low, inside a trust region that
grows on success and shrinks on failure.

Why this exists. Profiling the 756-task audit matrix showed every
optimizer in the portfolio is either a gradient method (27 optax
entries) or an evolution strategy (30 evosax entries), and at low budget
``randomsearch`` beats all of them on 130 of 716 tasks -- nothing in the
portfolio is sample-efficient. This fills that hole with a third kind of
search behaviour rather than a third variant of the two that exist.

Three choices are forced by the rollout harness rather than by the
algorithm, because ``UpdateClass.batch_run_fused`` runs optimizers under
``eqx.filter_vmap`` with a fixed-length scan, so every piece of state
must be a fixed-shape array:

surrogate without a polynomial tail
    The classical cubic RBF needs a linear tail, making the
    interpolation system ``(m + d + 1)``-square -- at ``d = 1024`` that
    is a 1089-square solve *per step*. A positive-definite kernel needs
    no tail, so the system is ``m``-square and independent of the
    dimensionality. Distances are averaged over coordinates for the same
    reason: it keeps the kernel lengthscale on the same scale at every
    ``d``.

fixed-capacity archive
    A model-based method conventionally keeps every point it has seen.
    Here the archive is a ring buffer of the most recent
    ``archive_size`` evaluations -- which is also the right locality for
    a trust-region model, since the recent points are the ones near the
    current centre.

candidate scoring instead of an inner optimizer
    Minimising the surrogate would normally call an optimizer inside the
    step. Nested optimization under ``vmap`` + ``scan`` is a poor fit, so
    this draws a fixed candidate set and scores it in one matmul --
    DYCORS' approach, which also gives the exploration term for free.

References
----------
Wild & Shoemaker, "Global convergence of radial basis function trust
region derivative-free algorithms", SIAM J. Optim. 21(3), 2011.
Regis & Shoemaker, "Combining radial basis function surrogates and
dynamic coordinate search in high-dimensional expensive black-box
optimization", Eng. Optim. 45(5), 2013.
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
from l2co_optimizers._src.core.popsize import resolve_popsize
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


#: Flop budget for the candidate-scoring step. Scoring costs
#: ``n_candidates * archive_size * num_dims`` and is the *only* part of a
#: step that grows with dimensionality -- the kernel solve is
#: ``archive_size``-cubed by construction. Holding the product fixed
#: keeps a step roughly as expensive at 1024 dimensions as at 2. The
#: value is chosen so the cap still applies up to 256 dimensions:
#: ``2**20 / (32 * 256) = 128``, and ``2**20 / (32 * 1024) = 32``.
CANDIDATE_FLOPS = 2**20

#: Bounds on the resolved candidate count. The floor keeps the surrogate
#: minimisation meaningful at extreme dimensionality; the ceiling is the
#: value that was previously hard-coded.
N_CANDIDATES_MIN = 32
N_CANDIDATES_MAX = 128


def default_n_candidates(num_dims: int, archive_size: int) -> int:
    """Candidates per step under a fixed scoring budget.

    A flat count is what made this optimizer unusable in high dimension:
    at 128 candidates a 1024-dimensional cell ran about an hour, and 45
    of them overran a 2.5-hour job limit during the evaluation campaign.
    Scaling the count down as the dimension rises keeps the per-step cost
    roughly constant instead.

    Parameters
    ----------
    num_dims : int
        Problem dimensionality.
    archive_size : int
        Ring-buffer capacity; scoring touches every archived point.

    Returns
    -------
    int
        Candidate count, clipped to
        ``[N_CANDIDATES_MIN, N_CANDIDATES_MAX]``.
    """
    n = CANDIDATE_FLOPS // max(archive_size * num_dims, 1)
    return int(min(max(n, N_CANDIDATES_MIN), N_CANDIDATES_MAX))


def rbf_popsize(dimensionality: int) -> int:
    """Evaluations per iteration; small, because the point is frugality.

    A trust-region model method conventionally evaluates one point per
    iteration. A small batch keeps that character while giving the
    harness a population to vectorize over.
    """
    del dimensionality
    return 5


def _mean_sq_dist(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-coordinate mean squared distance between two point sets.

    Averaging over coordinates rather than summing is what makes one
    kernel lengthscale work from 2 to 1024 dimensions.

    Parameters
    ----------
    a : jax.Array
        Points, shape ``(n, d)``.
    b : jax.Array
        Points, shape ``(m, d)``.

    Returns
    -------
    jax.Array
        Shape ``(n, m)``, non-negative.
    """
    d2 = (
        jnp.sum(a**2, axis=1)[:, None]
        + jnp.sum(b**2, axis=1)[None, :]
        - 2.0 * a @ b.T
    )
    return jnp.maximum(d2, 0.0) / a.shape[1]


def _unit(v: jax.Array) -> jax.Array:
    """Rescale to ``[0, 1]``; all-equal input maps to zeros."""
    lo, hi = jnp.min(v), jnp.max(v)
    return jnp.where(hi > lo, (v - lo) / jnp.maximum(hi - lo, 1e-12), 0.0)


#                                                                State & Params
# =============================================================================


@struct.dataclass
class State(BasePopulationState):
    """RBF trust-region state.

    Attributes
    ----------
    archive : jax.Array
        Ring buffer of evaluated points, shape ``(archive_size, d)``.
        Slots not yet written hold zeros and are masked out through
        ``archive_fitness``.
    archive_fitness : jax.Array
        Their objective values, shape ``(archive_size,)``; ``inf`` marks
        an unwritten slot.
    archive_index : jax.Array
        Next slot to overwrite (scalar int).
    center : jax.Array
        Trust-region centre -- the incumbent, shape ``(d,)``.
    center_fitness : jax.Array
        Objective at ``center`` (scalar).
    radius : jax.Array
        Trust-region radius (scalar), the std of the candidate
        perturbation and the kernel lengthscale.
    n_success : jax.Array
        Consecutive improving iterations (scalar int).
    n_failure : jax.Array
        Consecutive non-improving iterations (scalar int).
    """

    archive: jax.Array
    archive_fitness: jax.Array
    archive_index: jax.Array
    center: jax.Array
    center_fitness: jax.Array
    radius: jax.Array
    n_success: jax.Array
    n_failure: jax.Array


@struct.dataclass
class Params(BasePopulationParams):
    """RBF trust-region parameters (value-like; shapes live on the class).

    Attributes
    ----------
    radius_init, radius_min, radius_max : float
        Initial, smallest and largest trust-region radius. Reaching
        ``radius_min`` triggers a restart rather than a stall.
    expand, shrink : float
        Multipliers applied after ``success_tol`` consecutive
        improvements / ``failure_tol`` consecutive failures.
    success_tol, failure_tol : int
        How many consecutive outcomes it takes to move the radius.
    jitter : float
        Ridge added to the kernel matrix before solving.
    weight_period : int
        Length of the cycle over which the candidate score is swept from
        pure exploration to pure surrogate exploitation (DYCORS).
    perturb_prob : float
        Probability that a given coordinate is perturbed when drawing a
        candidate. Below 1 this is what makes the search usable in high
        dimension; one coordinate is always forced so no candidate
        duplicates the centre.
    x_min, x_max : float
        Box bounds; injected from the run's ``bounded``.
    """

    radius_init: float = 0.2
    radius_min: float = 1e-3
    radius_max: float = 1.6
    expand: float = 2.0
    shrink: float = 0.5
    success_tol: int = 3
    failure_tol: int = 5
    jitter: float = 1e-6
    weight_period: int = 5
    perturb_prob: float = 0.2
    x_min: float = -jnp.inf
    x_max: float = jnp.inf


#                                                                     Algorithm
# =============================================================================


class RBFTrustRegion(PopulationBasedAlgorithm):
    """Surrogate-guided trust-region search.

    Attributes
    ----------
    archive_size : int
        Ring-buffer capacity, and the order of the kernel solve done on
        every iteration. Cost is ``O(archive_size ** 3)`` per step, so
        this trades model quality against step cost directly.
    n_candidates : int
        Trial points drawn and scored per iteration. Only ``popsize`` of
        them are ever evaluated on the real objective; the rest cost one
        surrogate evaluation each. Resolved from the dimensionality by
        default -- see :func:`default_n_candidates`.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        archive_size: int = 32,
        n_candidates: int | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable | None = None,
    ):
        """Initialize the RBF trust-region searcher.

        Parameters
        ----------
        population_size : int
            Real objective evaluations per iteration.
        solution : Solution
            Solution PyTree template (ravelled internally).
        archive_size : int, optional
            Ring-buffer capacity, by default 32. Kept small on purpose:
            the kernel solve is cubic in it and runs every step.
        n_candidates : int or None, optional
            Trial points scored per iteration. ``None`` (the default)
            resolves it from the dimensionality through
            :func:`default_n_candidates`, holding the per-step scoring
            cost roughly constant; pass an int to override.
        fitness_shaping_fn, metrics_fn : Callable, optional
            EvoSax hooks.

        Raises
        ------
        ValueError
            ``archive_size`` is below ``population_size``, or
            ``n_candidates`` is below ``population_size``.
        """
        if metrics_fn is None:
            from evosax.algorithms.population_based.base import metrics_fn as m

            metrics_fn = m
        super().__init__(
            population_size, solution, fitness_shaping_fn, metrics_fn
        )
        if archive_size < population_size:
            raise ValueError(
                "archive_size must be at least population_size; got "
                f"{archive_size} < {population_size}."
            )
        self.archive_size = archive_size
        if n_candidates is None:
            n_candidates = default_n_candidates(self.num_dims, archive_size)
        if n_candidates < population_size:
            raise ValueError(
                "n_candidates must be at least population_size; got "
                f"{n_candidates} < {population_size}."
            )
        self.n_candidates = n_candidates

    @property
    def _default_params(self) -> Params:
        return Params()

    def _init(self, key: jax.Array, params: Params) -> State:
        return State(
            population=jnp.full(
                (self.population_size, self.num_dims), jnp.nan
            ),
            fitness=jnp.full(self.population_size, jnp.inf),
            archive=jnp.zeros((self.archive_size, self.num_dims)),
            archive_fitness=jnp.full(self.archive_size, jnp.inf),
            archive_index=jnp.array(0, dtype=int),
            center=jnp.zeros((self.num_dims,)),
            center_fitness=jnp.array(jnp.inf),
            radius=jnp.array(params.radius_init),
            n_success=jnp.array(0, dtype=int),
            n_failure=jnp.array(0, dtype=int),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.array(jnp.inf),
            generation_counter=0,
        )

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        key_pert, key_coord, key_mask = jr.split(key, 3)
        valid = jnp.isfinite(state.archive_fitness)
        n_valid = jnp.sum(valid)
        ell = jnp.maximum(state.radius, params.radius_min)

        # --- fit the surrogate on the archive -------------------------
        # Invalid rows are replaced by identity rows so the solve stays
        # well posed with an almost-empty archive; their coefficients
        # are then zeroed and never contribute to a prediction.
        d2 = _mean_sq_dist(state.archive, state.archive)
        kernel = jnp.exp(-d2 / (2.0 * ell**2))
        pair = valid[:, None] & valid[None, :]
        eye = jnp.eye(self.archive_size)
        kernel = jnp.where(pair, kernel, eye)
        f_valid = jnp.where(valid, state.archive_fitness, 0.0)
        f_mean = jnp.sum(f_valid) / jnp.maximum(n_valid, 1)
        rhs = jnp.where(valid, state.archive_fitness - f_mean, 0.0)
        coef = jnp.linalg.solve(kernel + params.jitter * eye, rhs)
        coef = jnp.where(valid, coef, 0.0)

        # --- draw candidates around the incumbent ---------------------
        noise = (
            jr.normal(key_pert, (self.n_candidates, self.num_dims))
            * state.radius
        )
        perturb = (
            jr.uniform(key_mask, (self.n_candidates, self.num_dims))
            < params.perturb_prob
        )
        # Force one coordinate so no candidate collapses onto the centre.
        forced = jr.randint(key_coord, (self.n_candidates,), 0, self.num_dims)
        perturb = perturb.at[jnp.arange(self.n_candidates), forced].set(True)
        cand = state.center + jnp.where(perturb, noise, 0.0)
        cand = jnp.clip(cand, params.x_min, params.x_max)

        # --- score: surrogate value against distance to the archive ---
        d2c = _mean_sq_dist(cand, state.archive)
        surrogate = f_mean + jnp.exp(-d2c / (2.0 * ell**2)) @ coef
        far = jnp.sqrt(jnp.min(jnp.where(valid[None, :], d2c, jnp.inf), 1))
        # Cycle the weight so the search alternates between exploiting
        # the model and covering unvisited ground (DYCORS).
        phase = jnp.mod(state.generation_counter, params.weight_period)
        w = (phase + 1.0) / params.weight_period
        empty = n_valid == 0
        score = jnp.where(
            empty,
            -_unit(far),
            w * _unit(surrogate) + (1.0 - w) * (1.0 - _unit(far)),
        )
        idx = jnp.argsort(score)[: self.population_size]
        return cand[idx], state

    def _tell(
        self,
        key: jax.Array,
        population: Population,
        fitness: Fitness,
        state: State,
        params: Params,
    ) -> State:
        f = jnp.where(jnp.isnan(fitness), jnp.inf, fitness)

        # --- archive the batch (ring buffer) --------------------------
        slots = jnp.mod(
            state.archive_index + jnp.arange(self.population_size),
            self.archive_size,
        )
        archive = state.archive.at[slots].set(population)
        archive_fitness = state.archive_fitness.at[slots].set(f)
        archive_index = jnp.mod(
            state.archive_index + self.population_size, self.archive_size
        )

        # --- move the centre only on a real improvement ---------------
        best = jnp.argmin(f)
        improved = f[best] < state.center_fitness
        center = jnp.where(improved, population[best], state.center)
        center_fitness = jnp.where(improved, f[best], state.center_fitness)

        # --- adapt the radius on consecutive outcomes -----------------
        n_success = jnp.where(improved, state.n_success + 1, 0)
        n_failure = jnp.where(improved, 0, state.n_failure + 1)
        grow = n_success >= params.success_tol
        cut = n_failure >= params.failure_tol
        radius = jnp.where(
            grow,
            state.radius * params.expand,
            jnp.where(cut, state.radius * params.shrink, state.radius),
        )
        # A collapsed trust region means the model has stopped helping
        # here: restart at the incumbent rather than stall at the floor.
        restart = radius < params.radius_min
        radius = jnp.where(restart, params.radius_init, radius)
        radius = jnp.minimum(radius, params.radius_max)
        n_success = jnp.where(grow, 0, n_success)
        n_failure = jnp.where(cut | restart, 0, n_failure)

        return state.replace(
            population=population,
            fitness=f,
            archive=archive,
            archive_fitness=archive_fitness,
            archive_index=archive_index,
            center=center,
            center_fitness=center_fitness,
            radius=radius,
            n_success=n_success,
            n_failure=n_failure,
        )


#                                                                       Factory
# =============================================================================


def rbf_trust_region_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    popsize: PopSize = rbf_popsize,
    bounded: tuple[float, float] | None = (None, None),
    stop_fn: StopFunction | None = None,
    archive_size: int = 32,
    n_candidates: int | None = None,
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for :class:`RBFTrustRegion`.

    Splits the hyperparameters into shape-affecting constructor
    arguments (``archive_size``, ``n_candidates``, ``popsize``) and
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
        Real evaluations per iteration; defaults to :func:`rbf_popsize`.
    bounded : tuple[float, float] or None, optional
        Box bounds applied after each step and injected into the
        algorithm so candidates are drawn inside them.
    stop_fn : StopFunction or None, optional
        Stopping function.
    archive_size : int, optional
        Ring-buffer capacity, by default 32.
    n_candidates : int or None, optional
        Trial points scored per iteration; ``None`` scales it with the
        dimensionality (see :func:`default_n_candidates`).
    **hyperparameters
        Override fields on :class:`Params`.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    popsize = resolve_popsize(popsize, model)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    optimizer = RBFTrustRegion(
        population_size=popsize,
        solution=params,
        archive_size=archive_size,
        n_candidates=n_candidates,
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

    # The trust-region radius is this optimizer's scale, and it is the
    # only piece of state that means anything to another algorithm: the
    # archive and its surrogate coefficients are specific to this model.
    read_fn, write_fn = build_transfer_fns(
        "rbf_trust_region", FAMILY_POPULATION, hyperparameters
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


rbf_trust_region_mapping = {
    "rbf_trust_region": Partial(rbf_trust_region_update)
}
