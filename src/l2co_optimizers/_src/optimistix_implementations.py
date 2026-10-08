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
  ``bfgs`` with ``linesearch="wolfe"`` forms it at every point, and
  records it at every point.
  Nelder--Mead records the post-step simplex and its losses, with NaN
  gradients.
* **A strong-Wolfe BFGS** (ADR 0005). ``bfgs`` takes
  ``linesearch="wolfe"`` for scipy's BFGS line search (Moré--Thuente,
  :mod:`more_thuente`). optimistix hands a search only the loss at the
  point it evaluated, so :class:`WolfeBFGS` overrides the solver's
  ``step`` to give the search the slope there as well. The default stays
  Armijo backtracking, so existing ``bfgs`` runs are unchanged.
* **Bounds** are applied by projecting inside the objective, so every
  point the loss sees, and every point the history records, lies in the
  box; the solver's own state keeps the unprojected iterate.
* **Stochastic or minibatched losses** get a fresh batch and key every
  step, as every other entry does. The line searches compare against a
  loss cached from the previous step, so on such a loss they compare two
  different functions; that is accepted, not corrected.
* **Two Nelder--Mead workarounds** for optimistix 0.1.0: the population
  is injected as the simplex with ``eqx.tree_at`` (its ``y0_simplex``
  option rejects every n > 1, and its default simplex is degenerate for
  n >= 3), and after every step the best/worst vectors are re-read from
  the updated simplex (upstream reads them from the pre-update one, so
  they go stale; uncorrected, 5-D Rosenbrock stalls). Neither changes
  what is evaluated or billed.
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
from jaxtyping import Array, Bool, Int, PRNGKeyArray, PyTree, Scalar

# Local
from l2co_optimizers._src import more_thuente
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


def _project(y: PyTree, bounded: tuple[float | None, float | None]) -> PyTree:
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


#                                                          Strong-Wolfe BFGS
# =============================================================================

#: The constants of the line search behind
#: ``scipy.optimize.minimize(method="BFGS")``: ``line_search_wolfe1``'s
#: ``c1``, ``c2`` and ``xtol``.
_WOLFE_CONSTANTS = {"ftol": 1e-4, "gtol": 0.9, "xtol": 1e-14}

#: Evaluations one search may take, as in scipy's
#: ``scalar_search_wolfe1``.
_WOLFE_MAX_EVALS = 100

#: The step bounds scipy's BFGS passes to its line search.
_WOLFE_STEP_BOUNDS = (1e-100, 1e100)


def _step_bounds(dtype: Any) -> tuple[float, float]:
    """scipy's step bounds, narrowed so a product of two steps is finite.

    In float64 they are scipy's own. In float32 ``1e100`` is infinite, so
    both bounds are pulled in to the square roots of the smallest normal
    and the largest finite number.
    """
    finfo = jnp.finfo(dtype)
    low, high = _WOLFE_STEP_BOUNDS
    return max(low, float(finfo.tiny) ** 0.5), min(
        high, float(finfo.max) ** 0.5
    )


def _tree_dot(a: PyTree, b: PyTree) -> Scalar:
    """The inner product of two pytrees of the same structure."""
    return sum(jax.tree.leaves(jax.tree.map(jnp.vdot, a, b)), jnp.array(0.0))


def _tree_outer(a: PyTree, b: PyTree) -> PyTree:
    """``a b^T`` as a pytree of pytrees, the layout of a pytree operator."""
    return jax.tree.map(
        lambda x: jax.tree.map(lambda z: jnp.tensordot(x, z, axes=0), b), a
    )


def _tree_where(pred: Bool[Array, ""], true: PyTree, false: PyTree) -> PyTree:
    """Select between two pytrees of the same structure, leaf by leaf."""
    return jax.tree.map(lambda t, f: jnp.where(pred, t, f), true, false)


class WolfeSearchState(eqx.Module):
    """Line-search state of :class:`WolfeBFGS`.

    ``f_eval`` and ``accept`` mirror :class:`RecordingSearch`'s state, so
    :func:`_gradient_solver_update` reads both solvers' history the same
    way.

    Attributes
    ----------
    line : more_thuente.MoreThuenteState
        The Moré--Thuente search in progress.
    stp : Scalar
        Step, along the current direction, of the point being evaluated.
    f_prev : Scalar
        Loss at the accepted point before the current one: scipy's
        ``old_old_fval``, which sets each search's first trial step.
    n_evals : Int[Array, ""]
        Evaluations the current search has made.
    idle : Bool[Array, ""]
        Whether the point being evaluated is the accepted point itself,
        because not even steepest descent goes downhill from it.
    f_eval : Scalar
        Loss at the point the last step evaluated.
    grad_eval : PyTree
        Gradient there.
    accept : Bool[Array, ""]
        Whether the last step accepted that point.
    """

    line: more_thuente.MoreThuenteState
    stp: Scalar
    f_prev: Scalar
    n_evals: Int[Array, ""]
    idle: Bool[Array, ""]
    f_eval: Scalar
    grad_eval: PyTree
    accept: Bool[Array, ""]


class WolfeSearch(eqx.Module):
    """Builds :class:`WolfeBFGS`'s initial search state.

    It has no settings: the constants are scipy's, in
    ``_WOLFE_CONSTANTS``. :meth:`WolfeBFGS.step` runs the search itself,
    because optimistix hands a search only the loss at the point it
    evaluated, and the strong Wolfe conditions also need the slope there.
    """

    def init(self, y: PyTree, f_info_struct: Any) -> WolfeSearchState:
        dtype = f_info_struct.f.dtype
        low, high = _step_bounds(dtype)
        nan = jnp.full((), jnp.nan, dtype)
        # A placeholder: the first step always starts a fresh search.
        line, _ = more_thuente.start(
            1.0,
            jnp.zeros((), dtype),
            -1.0,
            ftol=_WOLFE_CONSTANTS["ftol"],
            stpmin=low,
            stpmax=high,
        )
        return WolfeSearchState(
            line=line,
            stp=jnp.ones((), dtype),
            f_prev=nan,
            n_evals=jnp.array(0),
            idle=jnp.array(False),
            f_eval=nan,
            grad_eval=jax.tree.map(lambda x: jnp.full_like(x, jnp.nan), y),
            accept=jnp.array(False),
        )


class WolfeBFGS(optx.AbstractBFGS):
    """BFGS with scipy's strong-Wolfe line search, one evaluation a step.

    ``scipy.optimize.minimize(method="BFGS")`` made steppable (ADR 0005).
    Each :meth:`step` evaluates one point, value and gradient together,
    and feeds it to a Moré--Thuente search (:mod:`more_thuente`) with
    scipy's constants and first-trial-step rule. While scipy's search
    succeeds, the points evaluated are scipy's.

    Where scipy would stop, a run here goes on to its budget:

    * a search that fails -- a warning from the search, 100 evaluations,
      or a non-finite next step -- accepts its last point if that point
      lowered the loss, and otherwise resets the Hessian estimate to the
      identity and starts again from the accepted point;
    * a direction that does not go downhill is replaced by steepest
      descent; when even that does not go downhill (a zero or non-finite
      gradient), every step re-evaluates the accepted point;
    * a non-finite loss or slope at a trial halves the step towards the
      best step of the search, without updating the search.

    The estimate is updated whenever the curvature ``s^T y`` is positive.
    optimistix's own update skips any ``s^T y`` below the dtype's
    epsilon, which in float32 stops it from updating near any optimum.

    Attributes
    ----------
    rtol, atol, norm, verbose
        Unused: a run never asks the solver to terminate. Taken from
        :class:`optimistix.BFGS` so the solver reads like one.
    use_inverse : bool
        Whether to approximate the inverse Hessian rather than the
        Hessian.
    descent : optimistix.NewtonDescent
        The quasi-Newton direction, as in :class:`optimistix.BFGS`.
    search : WolfeSearch
        Builds the search state; see :meth:`step`.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: optx.NewtonDescent
    search: WolfeSearch
    verbose: Callable[..., None]

    def __init__(self, use_inverse: bool = True):
        stock = optx.BFGS(
            rtol=_UNUSED_TOL, atol=_UNUSED_TOL, use_inverse=use_inverse
        )
        self.rtol = stock.rtol
        self.atol = stock.atol
        self.norm = stock.norm
        self.use_inverse = use_inverse
        self.descent = stock.descent
        self.search = WolfeSearch()
        self.verbose = stock.verbose

    def _operator(self, f_info: Any) -> Any:
        return f_info.hessian_inv if self.use_inverse else f_info.hessian

    def _with_operator(self, f_info: Any, operator: Any) -> Any:
        return eqx.tree_at(self._operator, f_info, operator)

    def _reset(self, f_info: Any, y: PyTree) -> Any:
        """``f_info`` with its Hessian estimate reset to the identity.

        Only the operator's matrix changes, so the result has exactly
        ``f_info``'s structure and the two can be selected between.
        """
        identity, _ = self.init_hessian(y, f_info.f, f_info.grad)
        operator = self._operator(f_info)
        return self._with_operator(
            f_info,
            eqx.tree_at(
                lambda o: o.pytree, operator, self._operator(identity).pytree
            ),
        )

    def update_hessian(
        self,
        y: PyTree,
        y_eval: PyTree,
        f_info: Any,
        f_eval_info: optx.FunctionInfo.EvalGrad,
        hessian_update_state: None,
    ) -> tuple[Any, None]:
        """The BFGS update, applied whenever ``s^T y`` is positive.

        The formulas are :class:`optimistix.AbstractBFGS`'s; only the
        threshold differs (positive, not above the dtype's epsilon).
        """
        grad = f_eval_info.grad
        s = jax.tree.map(jnp.subtract, y_eval, y)
        d = jax.tree.map(jnp.subtract, grad, f_info.grad)
        inner = _tree_dot(d, s)
        operator = self._operator(f_info)
        if self.use_inverse:
            hd = operator.mv(d)
            scale = (inner + _tree_dot(d, hd)) / inner**2
            matrix = jax.tree.map(
                lambda m, ss, hs, sh: m + scale * ss - (hs + sh) / inner,
                operator.pytree,
                _tree_outer(s, s),
                _tree_outer(hd, s),
                _tree_outer(s, hd),
            )
        else:
            bs = operator.mv(s)
            curvature = _tree_dot(s, bs)
            matrix = jax.tree.map(
                lambda m, dd, bb: m + dd / inner - bb / curvature,
                operator.pytree,
                _tree_outer(d, d),
                _tree_outer(bs, bs),
            )
        updated = eqx.tree_at(lambda o: o.pytree, operator, matrix)
        operator = _tree_where(inner > 0, updated, operator)
        return type(f_info)(f_eval_info.f, grad, operator), None

    def step(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
    ) -> tuple[PyTree, Any, Any]:
        """Evaluate one point and advance the search it belongs to.

        The point is ``state.y_eval``, chosen by the previous step. On
        the first step it is the starting point, which is accepted
        unconditionally, as optimistix does.
        """
        search = state.search_state
        (f_eval, aux_eval), grad_eval = jax.value_and_grad(
            lambda _y: fn(_y, args), has_aux=True
        )(state.y_eval)
        low, high = _step_bounds(f_eval.dtype)
        bounds = {"stpmin": low, "stpmax": high}

        # 1. What this evaluation means for the search it belongs to.
        direction = jax.tree.map(jnp.negative, state.descent_state.newton)
        slope = _tree_dot(grad_eval, direction)
        line, stp_next, task = more_thuente.iterate(
            search.line,
            search.stp,
            f_eval,
            slope,
            **_WOLFE_CONSTANTS,
            **bounds,
        )
        n_evals = search.n_evals + 1
        forced = state.first_step | search.idle
        finite = jnp.isfinite(f_eval) & jnp.isfinite(slope)
        live = ~forced & (n_evals < _WOLFE_MAX_EVALS)
        converged = live & finite & (task == more_thuente.TASK_CONVERGENCE)
        continuing = (
            live
            & finite
            & (task == more_thuente.TASK_FG)
            & jnp.isfinite(stp_next)
        )
        retry = live & ~finite
        failed = ~forced & ~converged & ~continuing & ~retry
        rescued = failed & finite & (f_eval < state.f_info.f)
        accept = forced | converged | rescued
        restart = failed & ~rescued
        new_search = accept | restart

        # 2. The point the next search starts from, and its estimate.
        accepted, _ = self.update_hessian(
            y,
            state.y_eval,
            state.f_info,
            optx.FunctionInfo.EvalGrad(f_eval, grad_eval),
            None,
        )
        f_info = _tree_where(accept, accepted, state.f_info)
        f_info = _tree_where(restart, self._reset(f_info, y), f_info)
        y_new = _tree_where(accept, state.y_eval, y)
        aux = _tree_where(accept, aux_eval, state.aux)

        # 3. Its direction: quasi-Newton, or steepest descent when that
        # does not go downhill.
        def downhill(s):
            return jnp.isfinite(s) & (s < 0)

        quasi_newton = self.descent.query(y_new, f_info, state.descent_state)
        slope_qn = _tree_dot(
            f_info.grad, jax.tree.map(jnp.negative, quasi_newton.newton)
        )
        steepest_info = self._reset(f_info, y_new)
        steepest = self.descent.query(
            y_new, steepest_info, state.descent_state
        )
        slope_sd = _tree_dot(
            f_info.grad, jax.tree.map(jnp.negative, steepest.newton)
        )
        use_steepest = ~downhill(slope_qn)
        idle = new_search & use_steepest & ~downhill(slope_sd)
        f_info = _tree_where(new_search & use_steepest, steepest_info, f_info)
        descent_state = _tree_where(
            new_search,
            _tree_where(use_steepest, steepest, quasi_newton),
            state.descent_state,
        )
        slope0 = jnp.where(use_steepest, slope_sd, slope_qn)

        # 4. scipy's first trial step: min(1, 2.02 (f - f_prev) / slope0),
        # with f_prev = f + |g| / 2 on the very first search.
        f_prev = jnp.where(
            accept,
            jnp.where(
                state.first_step,
                f_eval + optx.two_norm(grad_eval) / 2,
                state.f_info.f,
            ),
            search.f_prev,
        )
        alpha1 = jnp.minimum(1.0, 1.01 * 2 * (f_info.f - f_prev) / slope0)
        alpha1 = jnp.where(jnp.isfinite(alpha1) & (alpha1 >= low), alpha1, 1.0)
        fresh_line, _ = more_thuente.start(
            alpha1,
            f_info.f,
            slope0,
            ftol=_WOLFE_CONSTANTS["ftol"],
            **bounds,
        )

        # 5. The next point to evaluate.
        halfway = search.line.stx + 0.5 * (search.stp - search.line.stx)
        stp = jnp.where(
            new_search,
            jnp.where(idle, 0.0, alpha1),
            jnp.where(continuing, stp_next, halfway),
        ).astype(f_eval.dtype)
        line = _tree_where(
            new_search,
            fresh_line,
            _tree_where(continuing, line, search.line),
        )
        step, _ = self.descent.step(stp, descent_state)
        y_eval = _tree_where(idle, y_new, jax.tree.map(jnp.add, y_new, step))

        search_state = WolfeSearchState(
            line=line,
            stp=stp,
            f_prev=f_prev,
            n_evals=jnp.where(new_search, 0, n_evals),
            idle=idle,
            f_eval=f_eval,
            grad_eval=grad_eval,
            accept=accept,
        )
        state = eqx.tree_at(
            lambda s: (
                s.first_step,
                s.y_eval,
                s.search_state,
                s.f_info,
                s.descent_state,
                s.num_accepted_steps,
            ),
            state,
            (
                jnp.array(False),
                y_eval,
                search_state,
                f_info,
                descent_state,
                state.num_accepted_steps + accept,
            ),
        )
        return y_new, state, aux


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
    trial_gradients: bool = False,
) -> UpdateClass:
    """Wrap a gradient-based optimistix ``solver`` into an ``UpdateClass``.

    ``solver``'s search state must carry ``f_eval`` and ``accept``, as
    :class:`RecordingSearch`'s and :class:`WolfeSearchState` do. The
    population is a single point (``popsize == 1``); each step bills one
    evaluation. With ``trial_gradients`` the history records the
    gradient at every evaluated point, from the search state's
    ``grad_eval``; otherwise only on accepted steps, NaN on rejected ones.
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
        if trial_gradients:
            grads = search_state.grad_eval
        else:
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
    linesearch: str = "armijo",
    decrease_factor: float | None = None,
    slope: float | None = None,
    step_init: float | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="bfgs"``.

    With ``linesearch="armijo"`` (the default), :class:`optimistix.BFGS`
    with its backtracking Armijo line search. With ``"wolfe"``,
    :class:`WolfeBFGS`: scipy's BFGS, with its strong-Wolfe
    (Moré--Thuente) line search, made steppable (ADR 0005). Either way a
    step evaluates one point.

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
    linesearch : str, optional
        ``"armijo"`` (default) or ``"wolfe"``. The Wolfe search has no
        settings: its constants are scipy's (``c1=1e-4``, ``c2=0.9``).
        It also records the gradient at every evaluated point, where the
        Armijo search records NaN on rejected trials.
    decrease_factor : float or None, optional
        Armijo only: backtracking rate, in ``(0, 1)``; ``None`` means
        0.5.
    slope : float or None, optional
        Armijo only: sufficient-decrease slope, in ``(0, 1)``; ``None``
        means 0.1.
    step_init : float or None, optional
        Armijo only: first trial step size of each line search, ``> 0``;
        ``None`` means 1.0.

    Returns
    -------
    UpdateClass
        Configured optimizer wrapper.

    Raises
    ------
    ValueError
        If ``linesearch`` is neither ``"armijo"`` nor ``"wolfe"``, or if
        an Armijo setting is given with ``"wolfe"``.
    """
    armijo = {
        "decrease_factor": decrease_factor,
        "slope": slope,
        "step_init": step_init,
    }
    if linesearch == "wolfe":
        given = sorted(k for k, v in armijo.items() if v is not None)
        if given:
            raise ValueError(
                f"{given} configure the Armijo line search; "
                "linesearch='wolfe' takes none of them."
            )
        solver = WolfeBFGS(use_inverse=use_inverse)
    elif linesearch == "armijo":
        solver = eqx.tree_at(
            lambda s: s.search,
            optx.BFGS(
                rtol=_UNUSED_TOL, atol=_UNUSED_TOL, use_inverse=use_inverse
            ),
            _armijo(
                0.5 if decrease_factor is None else decrease_factor,
                0.1 if slope is None else slope,
                1.0 if step_init is None else step_init,
            ),
        )
    else:
        raise ValueError(
            f"Unknown bfgs linesearch {linesearch!r}; expected 'armijo' "
            "or 'wolfe'."
        )
    return _gradient_solver_update(
        solver,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        trial_gradients=linesearch == "wolfe",
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
    Parameters are those of :func:`bfgs_update`, without ``linesearch``:
    ``decrease_factor``, ``slope`` and ``step_init`` default to 0.5, 0.1
    and 1.0.

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
    :func:`dfp_update`.

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


def _reindex_best_and_worst(state: PyTree) -> PyTree:
    """Make ``state.best`` / ``state.worst`` hold the vertices they name.

    optimistix 0.1.0 computes the best and worst *indices* from the
    updated simplex but reads the *vectors* at those indices out of the
    pre-update simplex, so after a vertex replacement or a shrink they
    can point at a vertex that no longer exists -- e.g. the best vector
    becomes the old worst vertex. The next step then compares against
    and shrinks toward the wrong point. Re-reading both vectors from the
    updated simplex restores what the indices (and the stored losses)
    already describe; it evaluates nothing.
    """
    _, _, best_index = state.best
    _, _, worst_index = state.worst

    def pick(index):
        return jax.tree.map(lambda x: x[index], state.simplex)

    return eqx.tree_at(
        lambda s: (s.best[1], s.worst[1]),
        state,
        (pick(best_index), pick(worst_index)),
    )


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
    hyperparameters. After every step the state's best and worst vectors
    are re-read from the updated simplex (:func:`_reindex_best_and_worst`),
    correcting an optimistix 0.1.0 indexing bug.

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
        new_state = _reindex_best_and_worst(new_state)

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
