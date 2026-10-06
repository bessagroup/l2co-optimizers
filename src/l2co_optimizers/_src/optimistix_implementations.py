"""
optimistix minimisers as ``UpdateClass`` factories (ADR 0001).

Four registry entries -- ``"bfgs"``, ``"dfp"``, ``"nonlinearcg"`` and
``"neldermead"`` -- each drive an :mod:`optimistix` minimiser through
its own ``init`` / ``step`` pair, one solver step per l2co step. The
solver is never asked to stop (``terminate`` is not called): a run
ends at its budget or when the ``UpdateClass``'s ``stop_fn`` fires,
exactly like every other entry.

The decisions behind the wrapping are recorded in
``docs/adr/0001-optimistix-minimisers-as-plain-run-entries.md``; in
brief:

* **Billing.** ``OptHistory.fevals`` counts the calls optimistix
  actually makes. A gradient solver makes exactly one (value and
  gradient) per step. Nelder--Mead makes ``n + 1`` on its first step,
  ``2`` on an ordinary step and ``n + 3`` on a shrink step (``n`` the
  dimensionality) -- more than the algorithm needs, because optimistix
  re-evaluates an accepted reflection and the best vertex on a shrink.
* **History.** A gradient solver records the point it *evaluated*
  (``y_eval``, rejected line-search trials included), the loss there,
  and the gradient there when the step was accepted -- NaN when it was
  rejected, since optimistix only forms the gradient on acceptance.
  Nelder--Mead records the post-step simplex and its losses, with NaN
  gradients.
* **Bounds** are applied by projecting inside the objective, so every
  point the loss sees, and every point the history records, lies in the
  box; the solver's own state keeps the unprojected iterate.
* **Stochastic or minibatched losses** get a fresh batch and key every
  step, as every other entry does. The line searches compare against a
  loss cached from the previous step, so on such a loss they compare two
  different functions; that is accepted, not corrected.
* **No switching.** None of the four has an ``optimizer_parts``
  representation: optimistix solvers evaluate the objective themselves,
  which fits neither ``GradientParts`` nor ``PopulationParts``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from jax.tree_util import Partial
from jaxtyping import Array, Bool, PRNGKeyArray, PyTree, Scalar

# Local
from l2co_optimizers._src.core.opt_history import OptHistory
from l2co_optimizers._src.core.popsize import count_parameters
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
)
from l2co_optimizers._src.core.typing import (
    InputParameters,
    LossFunction,
    StopFunction,
)
from l2co_optimizers._src.core.update_class import UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Tolerances every optimistix solver constructor requires. They only
#: feed ``solver.terminate``, which this module never calls, so their
#: value has no effect on a run.
_UNUSED_TOL = 1e-8

#: Name -> beta function for ``nonlinearcg``'s ``method`` hyperparameter.
NONLINEAR_CG_METHODS: dict[str, Callable] = {
    "polak_ribiere": optx.polak_ribiere,
    "fletcher_reeves": optx.fletcher_reeves,
    "hestenes_stiefel": optx.hestenes_stiefel,
    "dai_yuan": optx.dai_yuan,
}

#: The registry names this module provides, normalized.
OPTIMISTIX_OPTIMIZERS: frozenset[str] = frozenset(
    {"bfgs", "dfp", "nonlinearcg", "neldermead"}
)

#                                                              Shared helpers
# =============================================================================


def _project(
    y: PyTree, bounded: tuple[float | None, float | None]
) -> PyTree:
    """Clip every leaf of ``y`` into ``bounded``."""
    return jax.tree.map(lambda x: jnp.clip(x, *bounded), y)


def _objective(
    static: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    bounded: tuple[float | None, float | None],
) -> Callable[[PyTree, tuple[dict, PRNGKeyArray]], tuple[Array, None]]:
    """The ``fn(y, args) -> (loss, aux)`` optimistix minimises.

    ``args`` is ``(sample, key)``: the step's data batch and its PRNG
    key. ``y`` is projected into the box before the loss sees it, which
    is how ``bounded`` reaches a solver that keeps its iterate in its
    own state.
    """

    def fn(y: PyTree, args: tuple[dict, PRNGKeyArray]) -> tuple[Array, None]:
        sample, key = args
        model = eqx.combine(_project(y, bounded), static)
        if pass_rng:
            return loss_fn(model, key=key, **sample), None
        return loss_fn(model, **sample), None

    return fn


def _init_solver_state(
    solver: optx.AbstractMinimiser,
    fn: Callable,
    y: PyTree,
    sample: dict,
    key: PRNGKeyArray | None,
) -> PyTree:
    """``solver.init`` with the output structure read off ``fn``."""
    args = (sample, jr.key(0) if key is None else key)
    f_struct, aux_struct = jax.eval_shape(fn, y, args)
    return solver.init(fn, y, args, {}, f_struct, aux_struct, frozenset())


def _expand(tree: PyTree) -> PyTree:
    """Add a leading population axis of one."""
    return jax.tree.map(lambda x: jnp.expand_dims(x, 0), tree)


#                                                           Recording search
# =============================================================================


class _RecordingSearchState(eqx.Module):
    """State of :class:`RecordingSearch`."""

    inner: Any
    f_eval: Scalar
    accept: Bool[Array, ""]


class RecordingSearch(optx.AbstractSearch):
    """A search that delegates to ``inner`` and remembers its last call.

    optimistix's gradient solvers hand their search the loss at the point
    they just evaluated, and keep it only if the step is accepted. This
    wrapper stores that loss, and the accept decision, in its own state,
    so the history can record rejected trials without evaluating the
    loss a second time. It changes nothing about the search itself.

    Attributes
    ----------
    inner : optimistix.AbstractSearch
        The search that decides the step.
    """

    inner: optx.AbstractSearch

    def init(self, y: PyTree, f_info_struct: Any) -> _RecordingSearchState:
        return _RecordingSearchState(
            inner=self.inner.init(y, f_info_struct),
            f_eval=jnp.full(
                f_info_struct.f.shape, jnp.nan, f_info_struct.f.dtype
            ),
            accept=jnp.array(False),
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: PyTree,
        y_eval: PyTree,
        f_info: Any,
        f_eval_info: Any,
        state: _RecordingSearchState,
    ) -> tuple[Scalar, Bool[Array, ""], Any, _RecordingSearchState]:
        step_size, accept, result, inner = self.inner.step(
            first_step, y, y_eval, f_info, f_eval_info, state.inner
        )
        return (
            step_size,
            accept,
            result,
            _RecordingSearchState(
                inner=inner, f_eval=f_eval_info.f, accept=accept
            ),
        )


def _armijo(
    decrease_factor: float, slope: float, step_init: float
) -> RecordingSearch:
    """A recorded :class:`optimistix.BacktrackingArmijo`."""
    return RecordingSearch(
        optx.BacktrackingArmijo(
            decrease_factor=decrease_factor,
            slope=slope,
            step_init=step_init,
        )
    )


#                                                          Gradient solvers
# =============================================================================


class GradientSolverState(eqx.Module):
    """Optimizer state of a gradient-based optimistix entry.

    Attributes
    ----------
    y : PyTree
        The solver's last accepted iterate, unprojected. It is what the
        next ``solver.step`` is called with; the population l2co carries
        is its projection into the box.
    solver_state : PyTree
        The optimistix solver's own state.
    """

    y: PyTree
    solver_state: PyTree


def _gradient_solver_update(
    solver: optx.AbstractMinimiser,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None,
    stop_fn: StopFunction | None,
) -> UpdateClass:
    """Wrap a gradient-based optimistix ``solver`` into an ``UpdateClass``.

    ``solver``'s search must be a :class:`RecordingSearch`. The
    population is a single point (``popsize == 1``); each step bills one
    evaluation.
    """
    if bounded is None:
        bounded = (None, None)

    _, static = eqx.partition(model, eqx.is_inexact_array)
    fn = _objective(static, loss_fn, pass_rng, bounded)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray | None = None, **sample
    ) -> GradientSolverState:
        y = jax.tree.map(lambda x: x[0], params)
        return GradientSolverState(
            y=y,
            solver_state=_init_solver_state(solver, fn, y, sample, key),
        )

    def step_fn(
        carry: tuple[InputParameters, GradientSolverState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[
        tuple[InputParameters, GradientSolverState, PRNGKeyArray], OptHistory
    ]:
        _, opt_state, key = carry
        new_key, eval_key = jr.split(key)

        # The point this step evaluates was chosen by the previous step.
        y_eval = opt_state.solver_state.y_eval
        y, solver_state, _ = solver.step(
            fn,
            opt_state.y,
            (sample, eval_key),
            {},
            opt_state.solver_state,
            frozenset(),
        )

        search_state = solver_state.search_state
        # On acceptance ``f_info`` is the evaluation at ``y_eval``; on
        # rejection it still describes the previous accepted point.
        grads = jax.tree.map(
            lambda g: jnp.where(search_state.accept, g, jnp.nan),
            solver_state.f_info.grad,
        )

        history = OptHistory(
            loss=jnp.expand_dims(search_state.f_eval, 0),
            params=_expand(_project(y_eval, bounded)),
            grads=_expand(grads),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(1, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )

        new_params = _expand(_project(y, bounded))
        return (
            (new_params, GradientSolverState(y, solver_state), new_key),
            history,
        )

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=1,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_GRADIENT,
    )


def bfgs_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    use_inverse: bool = True,
    decrease_factor: float = 0.5,
    slope: float = 0.1,
    step_init: float = 1.0,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="bfgs"``.

    :class:`optimistix.BFGS` with its backtracking Armijo line search.

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss. Must
        be differentiable.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box the loss is evaluated in: the iterate is projected into it
        before every evaluation. ``None``, or ``None`` on one side,
        leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.
    use_inverse : bool, optional
        Approximate the inverse Hessian (the default) rather than the
        Hessian.
    decrease_factor : float, optional
        Line-search backtracking rate, in ``(0, 1)``.
    slope : float, optional
        Armijo sufficient-decrease slope, in ``(0, 1)``.
    step_init : float, optional
        First trial step size of each line search, ``> 0``.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    solver = eqx.tree_at(
        lambda s: s.search,
        optx.BFGS(rtol=_UNUSED_TOL, atol=_UNUSED_TOL, use_inverse=use_inverse),
        _armijo(decrease_factor, slope, step_init),
    )
    return _gradient_solver_update(
        solver,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def dfp_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    use_inverse: bool = True,
    decrease_factor: float = 0.5,
    slope: float = 0.1,
    step_init: float = 1.0,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="dfp"``.

    :class:`optimistix.DFP` with its backtracking Armijo line search.
    Parameters are those of :func:`bfgs_update`.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.
    """
    solver = eqx.tree_at(
        lambda s: s.search,
        optx.DFP(rtol=_UNUSED_TOL, atol=_UNUSED_TOL, use_inverse=use_inverse),
        _armijo(decrease_factor, slope, step_init),
    )
    return _gradient_solver_update(
        solver,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def nonlinearcg_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    method: str = "polak_ribiere",
    decrease_factor: float = 0.5,
    slope: float = 0.1,
    step_init: float = 1.0,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="nonlinearcg"``.

    :class:`optimistix.NonlinearCG` with a backtracking Armijo line
    search. Parameters other than ``method`` are those of
    :func:`bfgs_update`.

    Parameters
    ----------
    method : str, optional
        The beta rule: ``"polak_ribiere"`` (default),
        ``"fletcher_reeves"``, ``"hestenes_stiefel"`` or ``"dai_yuan"``.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.

    Raises
    ------
    ValueError
        If ``method`` is not one of the four.
    """
    if method not in NONLINEAR_CG_METHODS:
        raise ValueError(
            f"Unknown nonlinearcg method {method!r}; expected one of "
            f"{sorted(NONLINEAR_CG_METHODS)}."
        )
    solver = optx.NonlinearCG(
        rtol=_UNUSED_TOL,
        atol=_UNUSED_TOL,
        method=NONLINEAR_CG_METHODS[method],
        search=_armijo(decrease_factor, slope, step_init),
    )
    return _gradient_solver_update(
        solver,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


#                                                                Nelder--Mead
# =============================================================================


class NelderMeadUpdateClass(UpdateClass):
    """``UpdateClass`` that runs realizations one at a time.

    Under the fused driver the realization axis is vmapped, which turns
    optimistix's ``lax.cond`` into a select: the shrink branch -- a
    re-evaluation of the whole simplex -- would then be computed on
    every step. Mapping realizations sequentially keeps the branches
    real. Nothing else differs from :class:`UpdateClass`.
    """

    sequential_realizations: ClassVar[bool] = True


def neldermead_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="neldermead"``.

    :class:`optimistix.NelderMead` with the population as its simplex:
    ``popsize`` is the dimensionality plus one, and the initial
    population drawn by the sampler is the initial simplex. It takes no
    hyperparameters.

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss. Every
        evaluation within one step shares that step's key.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    opt_hash : int
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box the loss is evaluated in: vertices are projected into it
        before every evaluation. ``None``, or ``None`` on one side,
        leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper; its ``max_fevals_per_step`` is
        ``n + 3``, the cost of a shrink step.
    """
    if bounded is None:
        bounded = (None, None)

    n = count_parameters(model)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    fn = _objective(static, loss_fn, pass_rng, bounded)
    solver = optx.NelderMead(rtol=_UNUSED_TOL, atol=_UNUSED_TOL)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray | None = None, **sample
    ) -> PyTree:
        y = jax.tree.map(lambda x: x[0], params)
        state = _init_solver_state(solver, fn, y, sample, key)
        # optimistix 0.1.0's ``y0_simplex`` option rejects every simplex
        # with n > 1, so build the default state from one vertex and put
        # the population in its place. ``f_simplex`` is still all-inf and
        # ``first_pass`` true, so the first step evaluates the whole
        # injected simplex without moving it.
        return eqx.tree_at(lambda s: s.simplex, state, params)

    def step_fn(
        carry: tuple[InputParameters, PyTree, PRNGKeyArray],
        sample: dict,
    ) -> tuple[tuple[InputParameters, PyTree, PRNGKeyArray], OptHistory]:
        _, opt_state, key = carry
        new_key, eval_key = jr.split(key)

        _, new_state, _ = solver.step(
            fn,
            jax.tree.map(lambda x: x[0], opt_state.simplex),
            (sample, eval_key),
            {},
            opt_state,
            frozenset(),
        )

        # optimistix evaluates the whole simplex on its first pass, two
        # vertices on every later step, and the whole simplex again when
        # it shrinks.
        shrunk = new_state.stats.n_shrink - opt_state.stats.n_shrink
        fevals = jnp.where(
            opt_state.first_pass, n + 1, 2 + shrunk * (n + 1)
        ).astype(int)

        simplex = _project(new_state.simplex, bounded)
        history = OptHistory(
            loss=new_state.f_simplex,
            params=simplex,
            grads=jax.tree.map(lambda p: jnp.full_like(p, jnp.nan), simplex),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=fevals,
            iterations=jnp.array(1, dtype=int),
        )
        return (simplex, new_state, new_key), history

    return NelderMeadUpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=n + 1,
        hash=opt_hash,
        stop_fn=stop_fn,
        max_fevals_per_step=n + 3,
        family=FAMILY_POPULATION,
    )


optimistix_mapping = {
    "bfgs": Partial(bfgs_update),
    "dfp": Partial(dfp_update),
    "nonlinearcg": Partial(nonlinearcg_update),
    "neldermead": Partial(neldermead_update),
}
