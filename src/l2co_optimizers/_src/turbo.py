"""Trust-region Bayesian optimization (TuRBO).

Bayesian optimization fits a probabilistic model to what it has already
evaluated and spends the next evaluation where that model says the
payoff is best -- trading cheap model queries for expensive real ones.
Plain BO does this globally, which stops working in high dimension: the
posterior becomes flat, and the acquisition function has no opinion.
TuRBO restricts the whole business to a trust region that grows on
success and shrinks on failure, so the model only ever has to be right
locally.

Reference
---------
Eriksson, Pearce, Gardner, Turner & Poloczek, "Scalable Global
Optimization via Local Bayesian Optimization", NeurIPS 2019.

What separates this from :mod:`l2co_optimizers._src.rbf_trust_region`,
which is also a surrogate inside a trust region: this one models
*uncertainty*. The RBF searcher interpolates and falls back on distance
to the archive as a stand-in for "unexplored"; here the Gaussian-process
posterior gives a variance directly, and selection is by Thompson
sampling -- draw one plausible objective from the posterior and take its
minimiser -- so exploration is calibrated to what the model does not
know rather than to raw distance.

Deviations from the paper, all forced by the rollout harness running
optimizers under ``filter_vmap`` with a fixed-length scan:

no hyperparameter fitting
    TuRBO refits ARD lengthscales by marginal likelihood each step. That
    is an inner optimization, which nests badly under vmap and scan.
    Here the lengthscale is tied to the trust-region size, so the model
    sharpens as the region contracts. The trust region is therefore
    isotropic rather than stretched along the learned lengthscales.

marginal, not joint, Thompson sampling
    A joint posterior draw over ``n_candidates`` points needs their
    full covariance, at cubic cost in the candidate count. Sampling each
    candidate from its own marginal is linear and is what large-candidate
    implementations do in practice; it discards the correlation between
    candidates, which matters less when only the arg-min is used.

fixed-capacity archive
    The ring buffer of :mod:`rbf_trust_region`, for the same reason: a
    growing observation set has a growing shape. Capping it is also what
    keeps the Cholesky affordable, since that cost is cubic in the number
    of retained points and is paid every step.
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
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
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

from l2co_optimizers._src.evosax_implementations import (
    evosax_population_based_fn,
)
from l2co_optimizers._src.popsize import resolve_popsize

# Local
from l2co_optimizers._src.rbf_trust_region import (
    _mean_sq_dist,
    default_n_candidates,
)
from l2co_optimizers._src.state_transfer import (
    FAMILY_POPULATION,
    build_transfer_fns,
)
from l2co_optimizers._src.typing import PopSize, StopFunction, TaskLike
from l2co_optimizers._src.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Evaluations per iteration. BO exists to be frugal; a small batch keeps
#: that character while giving the harness something to vectorize over.
TURBO_POPSIZE = 5


def turbo_popsize(task: TaskLike) -> int:
    """Batch size, independent of the task."""
    del task
    return TURBO_POPSIZE


def _matern52(d2: jax.Array, ell: jax.Array) -> jax.Array:
    """Matern-5/2 covariance from per-coordinate mean squared distances.

    The standard BO kernel: twice differentiable, so it models a smooth
    objective without the squared exponential's assumption of infinite
    smoothness, which makes the posterior over-confident between points.

    Parameters
    ----------
    d2 : jax.Array
        Mean squared distances (see
        :func:`~l2co_optimizers._src.rbf_trust_region._mean_sq_dist`).
    ell : jax.Array
        Lengthscale, on the same per-coordinate scale.

    Returns
    -------
    jax.Array
        Covariances in ``(0, 1]``, same shape as ``d2``.
    """
    r = jnp.sqrt(jnp.maximum(d2, 1e-30)) / ell
    s = jnp.sqrt(5.0) * r
    return (1.0 + s + 5.0 * r**2 / 3.0) * jnp.exp(-s)


#                                                                State & Params
# =============================================================================


@struct.dataclass
class State(BasePopulationState):
    """TuRBO state.

    Attributes
    ----------
    archive : jax.Array
        Ring buffer of evaluated points, ``(archive_size, num_dims)``.
    archive_fitness : jax.Array
        Their objective values; ``inf`` marks an unwritten slot.
    archive_index : jax.Array
        Next slot to overwrite.
    center : jax.Array
        Trust-region centre -- the incumbent.
    center_fitness : jax.Array
        Objective at ``center``.
    length : jax.Array
        Trust-region side length, and the GP lengthscale with it.
    n_success, n_failure : jax.Array
        Consecutive improving / non-improving iterations.
    """

    archive: jax.Array
    archive_fitness: jax.Array
    archive_index: jax.Array
    center: jax.Array
    center_fitness: jax.Array
    length: jax.Array
    n_success: jax.Array
    n_failure: jax.Array


@struct.dataclass
class Params(BasePopulationParams):
    """TuRBO parameters (value-like; shapes live on the class).

    Attributes
    ----------
    length_init, length_min, length_max : float
        Trust-region side length: initial, the floor that triggers a
        restart, and the ceiling.
    expand, shrink : float
        Applied after ``success_tol`` successes / ``failure_tol``
        failures. The paper's doubling and halving.
    success_tol, failure_tol : int
        Consecutive outcomes needed to move the length.
    jitter : float
        Added to the kernel diagonal; stands in for observation noise
        and keeps the Cholesky well conditioned.
    perturb_prob : float
        Probability a coordinate is perturbed when drawing a candidate.
        Below one because a full-dimensional perturbation in high
        dimension lands nowhere near the incumbent.
    x_min, x_max : float
        Box bounds, injected from the run's ``bounded``.
    """

    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    expand: float = 2.0
    shrink: float = 0.5
    success_tol: int = 3
    failure_tol: int = 5
    jitter: float = 1e-6
    perturb_prob: float = 0.2
    x_min: float = -jnp.inf
    x_max: float = jnp.inf


#                                                                     Algorithm
# =============================================================================


class TuRBO(PopulationBasedAlgorithm):
    """Bayesian optimization confined to an adaptive trust region.

    Attributes
    ----------
    archive_size : int
        Observations retained. The Cholesky is cubic in this and is paid
        every step, so it buys model quality directly against step cost.
    n_candidates : int
        Points the posterior is sampled at per step. Only ``popsize`` are
        evaluated for real.
    """

    def __init__(
        self,
        population_size: int,
        solution: Solution,
        archive_size: int = 64,
        n_candidates: int | None = None,
        fitness_shaping_fn: Callable = identity_fitness_shaping_fn,
        metrics_fn: Callable | None = None,
    ):
        """Initialize TuRBO.

        Parameters
        ----------
        population_size : int
            Real evaluations per iteration.
        solution : Solution
            Solution PyTree template (ravelled internally).
        archive_size : int, optional
            Observations retained, by default 64.
        n_candidates : int or None, optional
            Posterior samples per step; ``None`` scales it with the
            dimensionality so the per-step cost stays roughly flat (see
            :func:`~l2co_optimizers._src.rbf_trust_region.default_n_candidates`).
        fitness_shaping_fn, metrics_fn : Callable, optional
            EvoSax hooks.

        Raises
        ------
        ValueError
            ``archive_size`` or the resolved ``n_candidates`` is below
            ``population_size``.
        """
        if metrics_fn is None:
            from evosax.algorithms.population_based.base import (
                metrics_fn as _m,
            )

            metrics_fn = _m
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
            length=jnp.array(params.length_init),
            n_success=jnp.array(0, dtype=int),
            n_failure=jnp.array(0, dtype=int),
            best_solution=jnp.full((self.num_dims,), jnp.nan),
            best_fitness=jnp.array(jnp.inf),
            generation_counter=0,
        )

    def _ask(
        self, key: jax.Array, state: State, params: Params
    ) -> tuple[Population, State]:
        key_pert, key_coord, key_mask, key_ts = jr.split(key, 4)
        valid = jnp.isfinite(state.archive_fitness)
        n_valid = jnp.sum(valid)
        ell = jnp.maximum(state.length, params.length_min)
        eye = jnp.eye(self.archive_size)

        # --- condition the GP on the archive --------------------------
        # Invalid slots get identity rows, so the factorization stays
        # well posed while the buffer fills; their weights come out zero
        # and they contribute nothing to a posterior.
        gram = _matern52(_mean_sq_dist(state.archive, state.archive), ell)
        pair = valid[:, None] & valid[None, :]
        gram = jnp.where(pair, gram, eye) + params.jitter * eye
        chol = jnp.linalg.cholesky(gram)

        # Standardize the targets so the prior variance is order one.
        f_valid = jnp.where(valid, state.archive_fitness, 0.0)
        f_mean = jnp.sum(f_valid) / jnp.maximum(n_valid, 1)
        centred = jnp.where(valid, state.archive_fitness - f_mean, 0.0)
        f_std = jnp.sqrt(jnp.sum(centred**2) / jnp.maximum(n_valid, 1)) + 1e-12
        y = centred / f_std
        alpha = jax.scipy.linalg.cho_solve((chol, True), y)

        # --- candidates inside the trust region -----------------------
        noise = (
            jr.normal(key_pert, (self.n_candidates, self.num_dims))
            * state.length
        )
        perturb = (
            jr.uniform(key_mask, (self.n_candidates, self.num_dims))
            < params.perturb_prob
        )
        forced = jr.randint(key_coord, (self.n_candidates,), 0, self.num_dims)
        perturb = perturb.at[jnp.arange(self.n_candidates), forced].set(True)
        cand = jnp.clip(
            state.center + jnp.where(perturb, noise, 0.0),
            params.x_min,
            params.x_max,
        )

        # --- posterior mean and variance at the candidates ------------
        k_star = jnp.where(
            valid[None, :],
            _matern52(_mean_sq_dist(cand, state.archive), ell),
            0.0,
        )
        mean = k_star @ alpha
        v = jax.scipy.linalg.solve_triangular(chol, k_star.T, lower=True)
        var = jnp.maximum(1.0 - jnp.sum(v**2, axis=0), 0.0)

        # --- Thompson sampling ----------------------------------------
        # One draw from each candidate's marginal posterior; the arg-min
        # is the pick. With no observations yet the posterior is the
        # prior, so this is a uniform draw over the candidates -- the
        # right behaviour on the first step, and no branch needed.
        draw = mean + jnp.sqrt(var) * jr.normal(key_ts, (self.n_candidates,))
        idx = jnp.argsort(draw)[: self.population_size]
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

        slots = jnp.mod(
            state.archive_index + jnp.arange(self.population_size),
            self.archive_size,
        )
        archive = state.archive.at[slots].set(population)
        archive_fitness = state.archive_fitness.at[slots].set(f)
        archive_index = jnp.mod(
            state.archive_index + self.population_size, self.archive_size
        )

        best = jnp.argmin(f)
        improved = f[best] < state.center_fitness
        center = jnp.where(improved, population[best], state.center)
        center_fitness = jnp.where(improved, f[best], state.center_fitness)

        n_success = jnp.where(improved, state.n_success + 1, 0)
        n_failure = jnp.where(improved, 0, state.n_failure + 1)
        grow = n_success >= params.success_tol
        cut = n_failure >= params.failure_tol
        length = jnp.where(
            grow,
            state.length * params.expand,
            jnp.where(cut, state.length * params.shrink, state.length),
        )
        # A collapsed region means the local model has stopped paying:
        # restart it rather than let the search freeze at the floor.
        restart = length < params.length_min
        length = jnp.minimum(
            jnp.where(restart, params.length_init, length), params.length_max
        )
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
            length=length,
            n_success=n_success,
            n_failure=n_failure,
        )


#                                                                       Factory
# =============================================================================


def turbo_update(
    task: TaskLike,
    opt_hash: int,
    popsize: PopSize = turbo_popsize,
    bounded: tuple[float, float] | None = (None, None),
    stop_fn: StopFunction | None = None,
    archive_size: int = 64,
    n_candidates: int | None = None,
    **hyperparameters,
) -> UpdateClass:
    """Construct an ``UpdateClass`` for :class:`TuRBO`.

    Splits the hyperparameters into shape-affecting constructor
    arguments (``archive_size``, ``n_candidates``, ``popsize``) and
    value-like :class:`Params` overrides, and injects the run's box
    bounds into ``Params.x_min`` / ``Params.x_max``.

    Parameters
    ----------
    task : TaskLike
        Task providing the model and loss function.
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    popsize : int or Callable[[TaskLike], int], optional
        Real evaluations per iteration; defaults to
        :func:`turbo_popsize`.
    bounded : tuple[float, float] or None, optional
        Box bounds applied after each step and injected so candidates
        are drawn inside them.
    stop_fn : StopFunction or None, optional
        Stopping function.
    archive_size : int, optional
        Observations retained, by default 64.
    n_candidates : int or None, optional
        Posterior samples per step; ``None`` scales with dimensionality.
    **hyperparameters
        Override fields on :class:`Params`.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    popsize = resolve_popsize(popsize, task)
    params, static = eqx.partition(task.model, eqx.is_inexact_array)

    optimizer = TuRBO(
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
        loss_fn=task.loss_fn,
        bounded=bounded,
        popsize=popsize,
        es_params=es_params,
        opt_hash=opt_hash,
        pass_rng=task.pass_rng,
    )

    # The trust-region length is the only piece of state another
    # algorithm can read as a scale; the observations and the posterior
    # built on them mean nothing outside this model.
    read_fn, write_fn = build_transfer_fns(
        "turbo", FAMILY_POPULATION, hyperparameters
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


turbo_mapping = {"turbo": Partial(turbo_update)}
