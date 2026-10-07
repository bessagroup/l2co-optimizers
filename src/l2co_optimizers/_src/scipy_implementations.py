"""
scipy.optimize minimisers as plain-run ``UpdateClass`` entries (ADR 0002).

Six registry entries -- ``"cobyqa"``, ``"powell"``, ``"tnc"``,
``"trustkrylov"``, ``"slsqp"`` and ``"trustconstr"`` -- each run
:func:`scipy.optimize.minimize` with the matching method. scipy owns
the optimization loop, calling the objective itself, so none of them
can be stepped from a JAX scan:
:class:`ScipyUpdateClass` overrides :meth:`~UpdateClass.run` to make
the *whole run* one :func:`jax.pure_callback`. On the host, the
callback drives scipy against a jitted evaluation of the loss and
returns fixed-length arrays the run loop turns into a
:class:`~l2co_optimizers.HistoryState`.

The decisions behind the wrapping are recorded in
``docs/adr/0002-scipy-minimisers-as-whole-run-callback-entries.md``, and
those for SLSQP and trust-constr in ADR 0003; in brief:

* **One iteration is one evaluation.** Every call scipy makes to the
  objective -- the value, or value and gradient together -- is one
  history entry, billed one evaluation, with population size one. A run
  makes exactly ``n_iterations`` of them.
* **The run lasts the whole budget.** Stopping tolerances are as tight as
  each method tolerates, so scipy stops only when the method cannot make
  progress. From then on every remaining iteration re-evaluates scipy's
  final point (``res.x``), really evaluating and billing it, as a
  converged optimistix solver keeps evaluating near its final point.
* **Failure is termination.** If scipy raises, or proposes a point that
  is not finite (never evaluated), the run stays at the best point
  evaluated so far for the rest of the budget. A NaN or infinite loss is
  passed to scipy unchanged.
* **trust-krylov's Hessian-vector products** are finite differences of
  JAX gradients, so each product is one more recorded, billed
  evaluation. TNC already works this way internally.
* **Stochastic or minibatched losses** get a fresh batch and key on
  every evaluation, drawn exactly as :meth:`UpdateClass.step` draws
  them for any other entry.
* **Bounds** are passed to scipy for COBYQA, Powell, TNC, SLSQP and
  trust-constr, which support them, with the starting point clipped into
  the box (trust-constr's box is ``keep_feasible``). trust-krylov does
  not, so its loss is evaluated at the point projected into the box,
  with the gradient taken through the projection. In all six, the
  history records the projected point.
* **A method that stops evaluating is stopped.** trust-constr can
  iterate forever without asking for an evaluation once its steps no
  longer move ``x``; after :data:`_STALL_ITERATIONS` such iterations the
  run ends at its current point, as if scipy had terminated.
* **No stopping criterion and no switching.** A ``stop_fn`` raises,
  since nothing can check it inside one callback, and
  ``optimizer_parts`` refuses all six.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.optimize as so
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.batching import BatchState
from l2co_optimizers._src.core.history_state import HistoryState
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_DERIVATIVE_FREE,
    FAMILY_GRADIENT,
)
from l2co_optimizers._src.core.typing import (
    InputParameters,
    LossFunction,
    StopFunction,
)
from l2co_optimizers._src.core.update_class import RunResult, UpdateClass

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

#: The registry names this module provides, normalized.
SCIPY_OPTIMIZERS: frozenset[str] = frozenset(
    {"cobyqa", "powell", "tnc", "trustkrylov", "slsqp", "trustconstr"}
)

#: An iteration or evaluation cap scipy never reaches: the budget is
#: enforced by the driver, which aborts scipy after ``n_iterations``
#: evaluations. The largest C ``int``, since TNC and trlib pass these
#: caps to C code that overflows on anything larger.
_UNREACHABLE = 2**31 - 1

#: Stopping tolerances, each as tight as its method tolerates (ADRs
#: 0002, 0003), so scipy ends a run only when the method cannot progress.
#: Chosen by running every method on sphere, a 1e6-conditioned
#: ellipsoid, Rosenbrock and Rastrigin at d = 2, 10, 40: at these values
#: COBYQA stops on its trust-region radius, Powell on "no improvement",
#: and TNC on a failed line search, all cleanly. trust-krylov has no
#: clean tolerance to find: whatever its ``gtol`` (1e-12, 1e-15 or 0),
#: it stops when its model stops predicting improvement, or proposes a
#: non-finite step at a machine-precision optimum, which the driver
#: treats as termination. SLSQP stops on "positive directional
#: derivative for linesearch" and trust-constr on its trust radius
#: falling below ``xtol``. trust-constr's ``xtol`` cannot be zero: its
#: radius reaches exactly zero and it then iterates forever without
#: evaluating anything.
_COBYQA_FINAL_TR_RADIUS = 1e-15
_POWELL_XTOL = 1e-15
_POWELL_FTOL = 0.0
_TNC_FTOL = 0.0
_TNC_XTOL = 0.0
_TNC_GTOL = 0.0
_TRUSTKRYLOV_GTOL = 0.0
_SLSQP_FTOL = 0.0
_TRUSTCONSTR_GTOL = 0.0
_TRUSTCONSTR_XTOL = 1e-15

#: Consecutive iterations without a new evaluation after which a method
#: counts as stopped (:meth:`_Driver.stalled`). A method whose steps no
#: longer change ``x`` -- a trust radius at zero, or a box it cannot
#: leave -- can otherwise iterate forever without evaluating anything,
#: and the run would never reach its budget.
_STALL_ITERATIONS = 1000

#: Gradients of recent non-probe evaluations kept for
#: :func:`_fd_hessp`; the iterate a Hessian-vector product is taken at
#: is always among the last few points scipy evaluated.
_GRAD_CACHE_SIZE = 8

_NO_STEP = (
    "A scipy entry runs its whole budget inside one host callback "
    "(ScipyUpdateClass.run); scipy owns the loop, so there is no single "
    "step to take. It runs only as a plain registry entry (ADR 0002)."
)

#                                                        Host-side driver
# =============================================================================


class _BudgetSpent(Exception):
    """Raised inside the objective once the budget is spent."""


class _NonFinitePoint(Exception):
    """Raised inside the objective when scipy proposes a non-finite point."""


#: ``evaluate(x, dataset, key, batch_state) -> (loss, grad, key,
#: batch_state)``, jitted; ``grad`` is ``None`` for a value-only method.
Evaluator = Callable[..., tuple[Any, Any, Any, BatchState]]

#: ``minimize(driver, x0, bounds) -> x`` -- runs one scipy method to
#: its own end and returns its final point. ``bounds`` is a
#: :class:`scipy.optimize.Bounds`, or ``None`` when there is no box.
Minimizer = Callable[["_Driver", np.ndarray, so.Bounds | None], np.ndarray]


def _np_bounds(
    bounded: tuple[float | None, float | None], n: int
) -> tuple[np.ndarray, np.ndarray]:
    """``bounded`` as two length-``n`` arrays, ``None`` as infinity."""
    lo, hi = bounded
    return (
        np.full(n, -np.inf if lo is None else lo),
        np.full(n, np.inf if hi is None else hi),
    )


class _Driver:
    """Hands scipy an objective and records every call it makes.

    One instance drives one realization. Every evaluation draws its
    batch and loss key exactly as :meth:`UpdateClass.step` does --
    ``batch_state.next(key)``, then ``new_key, eval_key = split(key)``
    -- so a stochastic or minibatched loss sees the same kind of
    sequence it would under any other entry.

    Attributes
    ----------
    losses : np.ndarray
        One loss per evaluation, NaN past :attr:`count`.
    count : int
        Evaluations made so far.
    best_x, best_loss
        The lowest-loss point evaluated so far (the earliest on ties;
        a NaN loss never wins), seeded with the starting point at an
        infinite loss.
    last_x : np.ndarray
        The last point evaluated, seeded with the starting point.
    batch_state : BatchState
        Advanced once per evaluation.
    eval_eps : float
        Machine epsilon of the dtype the loss is evaluated in (float32
        unless JAX runs in x64), which sets the finite-difference steps.
    """

    def __init__(
        self,
        evaluate: Evaluator,
        n_evals: int,
        x0: np.ndarray,
        dataset: dict[str, jax.Array],
        key: Any,
        batch_state: BatchState,
        bounded: tuple[float | None, float | None],
        eval_eps: float,
    ):
        self._evaluate = evaluate
        self.eval_eps = eval_eps
        self._dataset = dataset
        self._key = key
        self._lo, self._hi = _np_bounds(bounded, x0.size)
        self.batch_state = batch_state
        self.losses = np.full(n_evals, np.nan)
        self.count = 0
        self.best_x = x0
        self.best_loss = np.inf
        self.last_x = x0
        self._grads: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._count_at_iteration = 0
        self._iterations_without_eval = 0

    @property
    def budget_left(self) -> bool:
        return self.count < self.losses.size

    def _call(
        self, x: np.ndarray, cache_grad: bool = True
    ) -> tuple[float, np.ndarray | None]:
        """Evaluate at ``x``, record it and bill it."""
        if not self.budget_left:
            raise _BudgetSpent
        if not np.all(np.isfinite(x)):
            raise _NonFinitePoint
        f, g, self._key, self.batch_state = self._evaluate(
            jnp.asarray(x), self._dataset, self._key, self.batch_state
        )
        f = float(f)
        evaluated = np.clip(x, self._lo, self._hi)
        self.losses[self.count] = f
        self.count += 1
        self.last_x = evaluated
        if f < self.best_loss:
            self.best_loss, self.best_x = f, evaluated
        if g is not None:
            g = np.asarray(g, dtype=np.float64)
            if cache_grad:
                self._grads[x.tobytes()] = g
                while len(self._grads) > _GRAD_CACHE_SIZE:
                    self._grads.popitem(last=False)
        if not self.budget_left:
            raise _BudgetSpent
        return f, g

    def value(self, x: np.ndarray) -> float:
        """Objective for a derivative-free method."""
        return self._call(np.asarray(x, dtype=np.float64))[0]

    def value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective for a gradient method (``jac=True``)."""
        return self._call(np.asarray(x, dtype=np.float64))

    def probe_grad(self, x: np.ndarray) -> np.ndarray:
        """Gradient at a finite-difference probe point; not cached."""
        return self._call(np.asarray(x, dtype=np.float64), cache_grad=False)[1]

    def grad_at(self, x: np.ndarray) -> np.ndarray:
        """The gradient at ``x``: cached if scipy evaluated ``x`` lately.

        Only a cache miss evaluates (and bills) anything.
        """
        x = np.asarray(x, dtype=np.float64)
        cached = self._grads.get(x.tobytes())
        if cached is not None:
            return cached
        return self.value_and_grad(x)[1]

    def stalled(self) -> bool:
        """Call once per method iteration: has the method stopped evaluating?

        True once :data:`_STALL_ITERATIONS` consecutive calls passed
        without a new evaluation in between.
        """
        if self.count != self._count_at_iteration:
            self._count_at_iteration = self.count
            self._iterations_without_eval = 0
            return False
        self._iterations_without_eval += 1
        return self._iterations_without_eval >= _STALL_ITERATIONS

    def idle(self, x: np.ndarray) -> None:
        """Re-evaluate ``x`` until the budget is spent."""
        try:
            while self.budget_left:
                self._call(x)
        except _BudgetSpent:
            pass


def _fd_hessp(driver: _Driver) -> Callable[[np.ndarray, np.ndarray], Any]:
    """Finite-difference Hessian-vector products from JAX gradients.

    ``hessp(x, p) = (g(x + eps p) - g(x)) / eps``. The gradient at the
    probe point ``x + eps p`` is a real evaluation, recorded and billed
    like any other; the one at ``x`` is the evaluation scipy already
    made there. ``eps`` is the usual forward-difference step, scaled so
    that ``eps * |p|`` is ``sqrt(eval_eps) * max(1, |x|)``, with
    ``eval_eps`` the machine epsilon of the dtype the loss is evaluated
    in -- a float64-sized step would vanish under float32 rounding.
    """
    root_eps = float(np.sqrt(driver.eval_eps))

    def hessp(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        p_norm = float(np.linalg.norm(p))
        if p_norm == 0.0:
            return np.zeros_like(x)
        eps = root_eps * max(1.0, float(np.linalg.norm(x))) / p_norm
        g0 = driver.grad_at(x)
        g1 = driver.probe_grad(x + eps * p)
        return (g1 - g0) / eps

    return hessp


def _drive(
    minimize: Minimizer,
    evaluate: Evaluator,
    bounded: tuple[float | None, float | None],
    n_evals: int,
    key_impl: Any,
    x0: np.ndarray,
    key_data: np.ndarray,
    perm: np.ndarray,
    pos: np.ndarray,
    dataset: dict[str, np.ndarray],
    *,
    batch_size: int,
    dataset_size: int,
) -> tuple[np.ndarray, ...]:
    """One realization, on the host: scipy, then idle to the budget.

    Returns ``(losses, best_x, best_loss, last_x, perm, pos)`` in
    float64 / the batch state's own dtypes.
    """
    key = (
        jnp.asarray(key_data)
        if key_impl is None
        else jr.wrap_key_data(jnp.asarray(key_data), impl=key_impl)
    )
    eval_eps = float(np.finfo(np.asarray(x0).dtype).eps)
    lo, hi = _np_bounds(bounded, np.size(x0))
    x0 = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    driver = _Driver(
        evaluate=evaluate,
        n_evals=n_evals,
        x0=x0,
        dataset=jax.tree.map(jnp.asarray, dataset),
        key=key,
        batch_state=BatchState(
            perm=jnp.asarray(perm),
            pos=jnp.asarray(pos),
            batch_size=batch_size,
            dataset_size=dataset_size,
        ),
        bounded=bounded,
        eval_eps=eval_eps,
    )
    bounds = None if bounded == (None, None) else so.Bounds(lo, hi)

    x_final: np.ndarray | None
    try:
        x_final = np.asarray(minimize(driver, x0, bounds), dtype=np.float64)
        if np.all(np.isfinite(x_final)):
            reason = "scipy terminated"
        else:
            x_final, reason = driver.best_x, "scipy returned a non-finite x"
    except _BudgetSpent:
        x_final, reason = None, "budget spent"
    except _NonFinitePoint:
        x_final, reason = driver.best_x, "scipy proposed a non-finite x"
    except Exception as error:  # noqa: BLE001 -- any scipy failure ends it
        x_final, reason = driver.best_x, f"scipy raised {error!r}"
    logger.debug(
        "scipy stopped after %d of %d evaluations: %s",
        driver.count,
        n_evals,
        reason,
    )
    if x_final is not None:
        driver.idle(x_final)

    return (
        driver.losses,
        driver.best_x,
        np.float64(driver.best_loss),
        driver.last_x,
        np.asarray(driver.batch_state.perm),
        np.asarray(driver.batch_state.pos),
    )


#                                                             UpdateClass
# =============================================================================


class ScipyUpdateClass(UpdateClass):
    """``UpdateClass`` that runs a scipy minimiser as one host callback.

    :meth:`run` hands the whole budget to scipy in a single
    :func:`jax.pure_callback`, and :meth:`~UpdateClass.batch_run` maps
    that over realizations one at a time
    (:attr:`sequential_realizations`). There is no per-step form:
    ``init_fn`` returns a scalar placeholder (it is called through the
    inherited :meth:`~UpdateClass.init_state`), and :meth:`step` raises,
    because scipy owns the loop.

    Attributes
    ----------
    driver : Callable
        ``driver(n_evals, key_impl, x0, key_data, perm, pos, dataset, *,
        batch_size, dataset_size) -> (losses, best_x, best_loss, last_x,
        perm, pos)``: the host side of one realization, with the scipy
        method, the jitted loss and ``bounded`` already bound.
    """

    sequential_realizations: ClassVar[bool] = True

    driver: Callable = eqx.field(static=True)

    def __init__(self, driver: Callable, hash: int, family: str):
        """Initialise the scipy update class.

        Parameters
        ----------
        driver : Callable
            The host-side driver; see the class docstring.
        hash : int
            Optimizer hash stamped into ``HistoryState.update_step``.
        family : str
            Which optimizer family this is.
        """

        def _placeholder_init_fn(
            params: InputParameters,
            key: PRNGKeyArray | None = None,
            **sample: Any,
        ) -> jax.Array:
            return jnp.array(0, dtype=int)

        def _no_step_fn(carry, sample):
            raise NotImplementedError(_NO_STEP)

        super().__init__(
            init_fn=_placeholder_init_fn,
            step_fn=_no_step_fn,
            popsize=1,
            hash=hash,
            family=family,
        )
        self.driver = driver

    def step(self, carry, xs, dataset):
        """Not available: scipy owns the loop, so there is no single step.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(_NO_STEP)

    def run(
        self,
        opt_state: PyTree,
        params: InputParameters,
        batch_state: BatchState,
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        n_iterations: int,
        verbose: bool,
    ) -> RunResult:
        """Run scipy for ``n_iterations`` evaluations in one host callback.

        Parameters
        ----------
        opt_state : PyTree
            The scalar placeholder from ``init_fn``; returned unchanged.
        params : InputParameters
            Starting population, shape ``(1, ...)``: its one member is
            scipy's starting point.
        batch_state : BatchState
            Batch state each evaluation draws its batch from; advanced
            once per evaluation.
        dataset : dict[str, jax.Array]
            Dataset for evaluation.
        key : PRNGKeyArray
            PRNG key; split once per evaluation, as :meth:`step` would.
        n_iterations : int
            Number of evaluations, each one history entry.
        verbose : bool
            Ignored.

        Returns
        -------
        RunResult
            ``(params, best_params, best_loss, opt_state, batch_state,
            history_state)``, as :meth:`UpdateClass.run`. ``params`` is
            the last evaluated point, with a population axis of one.
        """
        del verbose
        x0, unravel = ravel_pytree(jax.tree.map(lambda p: p[0], params))
        dtype = x0.dtype
        n = x0.shape[0]

        typed_key = jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
        key_impl = jr.key_impl(key) if typed_key else None
        key_data = jr.key_data(key) if typed_key else key

        result_shapes = (
            jax.ShapeDtypeStruct((n_iterations,), dtype),
            jax.ShapeDtypeStruct((n,), dtype),
            jax.ShapeDtypeStruct((), dtype),
            jax.ShapeDtypeStruct((n,), dtype),
            jax.ShapeDtypeStruct(
                batch_state.perm.shape, batch_state.perm.dtype
            ),
            jax.ShapeDtypeStruct(batch_state.pos.shape, batch_state.pos.dtype),
        )

        def host(*operands):
            out = self.driver(
                n_iterations,
                key_impl,
                *operands,
                batch_size=batch_state.batch_size,
                dataset_size=batch_state.dataset_size,
            )
            return tuple(
                np.asarray(o, dtype=s.dtype)
                for o, s in zip(out, result_shapes, strict=True)
            )

        losses, best_x, best_loss, last_x, perm, pos = jax.pure_callback(
            host,
            result_shapes,
            x0,
            key_data,
            batch_state.perm,
            batch_state.pos,
            dataset,
            vmap_method="sequential",
        )

        history_state = HistoryState.from_reduced(
            losses,
            losses,
            jnp.zeros_like(losses),
            jnp.full((n_iterations,), self.hash, dtype=int),
            jnp.ones((n_iterations,), dtype=int),
            jnp.ones((n_iterations,), dtype=int),
        )
        batch_state = eqx.tree_at(
            lambda b: (b.perm, b.pos), batch_state, (perm, pos)
        )
        return (
            jax.tree.map(lambda p: jnp.expand_dims(p, 0), unravel(last_x)),
            unravel(best_x),
            best_loss,
            opt_state,
            batch_state,
            history_state,
        )


#                                                                Factories
# =============================================================================


def _make_evaluator(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    bounded: tuple[float | None, float | None],
    with_grad: bool,
) -> Evaluator:
    """The jitted per-evaluation function the host driver calls.

    It mirrors one :meth:`UpdateClass.step`: draw the batch from
    ``batch_state`` with the carried key, split the key, and evaluate
    the loss (and its gradient) at the point projected into ``bounded``,
    with the gradient taken through the projection.
    """
    model_params, static = eqx.partition(model, eqx.is_inexact_array)
    _, unravel = ravel_pytree(model_params)

    def loss_at(
        x: Float[Array, " n"], sample: dict, key: PRNGKeyArray
    ) -> Float[Array, ""]:
        model_ = eqx.combine(unravel(jnp.clip(x, *bounded)), static)
        if pass_rng:
            return loss_fn(model_, key=key, **sample)
        return loss_fn(model_, **sample)

    @jax.jit
    def evaluate(
        x: Float[Array, " n"],
        dataset: dict[str, jax.Array],
        key: PRNGKeyArray,
        batch_state: BatchState,
    ):
        batch_idxs, batch_state = batch_state.next(key)
        sample = jax.tree.map(lambda a: a[batch_idxs], dataset)
        new_key, eval_key = jr.split(key)
        if with_grad:
            f, g = jax.value_and_grad(loss_at)(x, sample, eval_key)
        else:
            f, g = loss_at(x, sample, eval_key), None
        return f, g, new_key, batch_state

    return evaluate


def _scipy_update(
    minimize: Minimizer,
    *,
    name: str,
    family: str,
    with_grad: bool,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None,
    stop_fn: StopFunction | None,
) -> UpdateClass:
    """Wrap a scipy ``minimize`` call into a :class:`ScipyUpdateClass`."""
    if stop_fn is not None:
        raise ValueError(
            f"{name!r} cannot honour a stopping criterion: it runs its "
            f"whole budget inside one scipy call, where nothing can check "
            f"a stop_fn between steps. Remove the stopping criteria from "
            f"this OptimizationStep (ADR 0002)."
        )
    bounded = (None, None) if bounded is None else tuple(bounded)
    evaluate = _make_evaluator(
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        bounded=bounded,
        with_grad=with_grad,
    )
    return ScipyUpdateClass(
        driver=functools.partial(_drive, minimize, evaluate, bounded),
        hash=opt_hash,
        family=family,
    )


def cobyqa_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    initial_tr_radius: float = 1.0,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="cobyqa"``.

    ``scipy.optimize.minimize(method="COBYQA")``: a derivative-free
    trust-region method on quadratic interpolation models (Ragonneau &
    Zhang). Its cost per step grows steeply with the dimensionality.

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
        Stable hash stamped into ``HistoryState.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box handed to COBYQA, which never evaluates outside it.
        ``None``, or ``None`` on one side, leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Must be ``None``; see Raises.
    initial_tr_radius : float, optional
        Initial trust-region radius; scipy's default ``1.0``.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value,
            x0,
            method="COBYQA",
            bounds=bounds,
            options=dict(
                initial_tr_radius=initial_tr_radius,
                final_tr_radius=_COBYQA_FINAL_TR_RADIUS,
                maxfev=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        ).x

    return _scipy_update(
        minimize,
        name="cobyqa",
        family=FAMILY_DERIVATIVE_FREE,
        with_grad=False,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def powell_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="powell"``.

    ``scipy.optimize.minimize(method="Powell")``: Powell's conjugate
    direction method, a sequence of derivative-free line searches. It
    takes no hyperparameters. Parameters are those of
    :func:`cobyqa_update`, without ``initial_tr_radius``; with a box,
    Powell's line searches stay inside it.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value,
            x0,
            method="Powell",
            bounds=bounds,
            options=dict(
                xtol=_POWELL_XTOL,
                ftol=_POWELL_FTOL,
                maxfev=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        ).x

    return _scipy_update(
        minimize,
        name="powell",
        family=FAMILY_DERIVATIVE_FREE,
        with_grad=False,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def tnc_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    eta: float = 0.25,
    stepmx: float = 10.0,
    max_cg_iterations: int | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="tnc"``.

    ``scipy.optimize.minimize(method="TNC")``: truncated Newton (Nash),
    a line-search Newton method whose Newton step is solved by
    conjugate gradients on finite-difference Hessian-vector products.
    Every gradient TNC asks for, including those inside its
    finite differences, is one evaluation. Parameters other than the
    three below are those of :func:`cobyqa_update`; with a box, TNC
    keeps every iterate inside it.

    Parameters
    ----------
    eta : float, optional
        Severity of the line search, in ``[0, 1]``; scipy's default
        ``0.25``.
    stepmx : float, optional
        Maximum step of the line search (TNC may raise it); scipy's
        default ``10.0``.
    max_cg_iterations : int or None, optional
        Hessian-vector products per Newton step (scipy's ``maxCGit``).
        ``None`` (the default) is scipy's ``max(1, min(50, n / 2))``.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value_and_grad,
            x0,
            jac=True,
            method="TNC",
            bounds=bounds,
            options=dict(
                eta=eta,
                stepmx=stepmx,
                maxCGit=-1 if max_cg_iterations is None else max_cg_iterations,
                # TNC's own finite differences, sized to the precision the
                # loss is evaluated in; float64 resolves to TNC's default.
                accuracy=driver.eval_eps,
                ftol=_TNC_FTOL,
                xtol=_TNC_XTOL,
                gtol=_TNC_GTOL,
                maxfun=_UNREACHABLE,
            ),
        ).x

    return _scipy_update(
        minimize,
        name="tnc",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def trustkrylov_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    initial_trust_radius: float = 1.0,
    max_trust_radius: float = 1000.0,
    eta: float = 0.15,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="trustkrylov"``.

    ``scipy.optimize.minimize(method="trust-krylov")``: a Newton
    trust-region method whose subproblem is solved in a Krylov space
    (GLTR, via trlib). Its Hessian-vector products are finite
    differences of JAX gradients, each one evaluation. Parameters other
    than the three below are those of :func:`cobyqa_update`, except
    that trust-krylov takes no box: its loss is evaluated at the point
    projected into ``bounded``, so a coordinate that leaves the box sees
    a zero gradient and stays where it is.

    Parameters
    ----------
    initial_trust_radius : float, optional
        Initial trust-region radius; scipy's default ``1.0``.
    max_trust_radius : float, optional
        Largest trust-region radius; scipy's default ``1000.0``.
    eta : float, optional
        Acceptance threshold on the ratio of actual to predicted
        reduction, in ``[0, 0.25)``; scipy's default ``0.15``.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        del bounds  # trust-krylov takes none; projected in the loss
        return so.minimize(
            driver.value_and_grad,
            x0,
            jac=True,
            hessp=_fd_hessp(driver),
            method="trust-krylov",
            options=dict(
                initial_trust_radius=initial_trust_radius,
                max_trust_radius=max_trust_radius,
                eta=eta,
                gtol=_TRUSTKRYLOV_GTOL,
                maxiter=_UNREACHABLE,
            ),
        ).x

    return _scipy_update(
        minimize,
        name="trustkrylov",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def slsqp_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="slsqp"``.

    ``scipy.optimize.minimize(method="SLSQP")``: Kraft's sequential
    least-squares quadratic programming, an SQP method with a dense BFGS
    Hessian and an L1 merit line search. With no constraints beyond a
    box, each step is a bound-constrained least-squares subproblem. It
    takes no hyperparameters. Its dense matrices make the cost per step
    grow steeply above about a thousand dimensions. Parameters are those
    of :func:`cobyqa_update`, without ``initial_tr_radius``; with a box,
    SLSQP keeps every iterate inside it.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value_and_grad,
            x0,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            options=dict(ftol=_SLSQP_FTOL, maxiter=_UNREACHABLE),
        ).x

    return _scipy_update(
        minimize,
        name="slsqp",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def trustconstr_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    initial_tr_radius: float = 1.0,
    initial_barrier_parameter: float = 0.1,
    initial_barrier_tolerance: float = 0.1,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="trustconstr"``.

    ``scipy.optimize.minimize(method="trust-constr")`` with scipy's
    default dense BFGS Hessian approximation. Without a box it is
    Byrd-Omojokun trust-region SQP, which on an unconstrained problem is
    a quasi-Newton trust region solved by projected conjugate gradients.
    With a box it is a trust-region interior-point (barrier) method, and
    the box is ``keep_feasible``, so every iterate stays inside it.
    Parameters other than the three below are those of
    :func:`cobyqa_update`. Its pure-Python iteration makes it slow, and
    its dense Hessian makes the cost per step grow steeply above about a
    thousand dimensions.

    Parameters
    ----------
    initial_tr_radius : float, optional
        Initial trust-region radius; scipy's default ``1.0``.
    initial_barrier_parameter : float, optional
        Initial barrier parameter of the interior-point method; scipy's
        default ``0.1``. Used only with a box.
    initial_barrier_tolerance : float, optional
        Initial tolerance of the barrier subproblem; scipy's default
        ``0.1``. Used only with a box.

    Returns
    -------
    UpdateClass
        A :class:`ScipyUpdateClass`.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``.
    """

    def minimize(driver, x0, bounds):
        if bounds is not None:
            bounds = so.Bounds(bounds.lb, bounds.ub, keep_feasible=True)

        def callback(intermediate_result):
            if driver.stalled():
                raise StopIteration

        return so.minimize(
            driver.value_and_grad,
            x0,
            jac=True,
            hess=so.BFGS(),
            method="trust-constr",
            bounds=bounds,
            callback=callback,
            options=dict(
                initial_tr_radius=initial_tr_radius,
                initial_barrier_parameter=initial_barrier_parameter,
                initial_barrier_tolerance=initial_barrier_tolerance,
                gtol=_TRUSTCONSTR_GTOL,
                xtol=_TRUSTCONSTR_XTOL,
                maxiter=_UNREACHABLE,
            ),
        ).x

    return _scipy_update(
        minimize,
        name="trustconstr",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


scipy_mapping = {
    "cobyqa": Partial(cobyqa_update),
    "powell": Partial(powell_update),
    "tnc": Partial(tnc_update),
    "trustkrylov": Partial(trustkrylov_update),
    "slsqp": Partial(slsqp_update),
    "trustconstr": Partial(trustconstr_update),
}
