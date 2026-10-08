"""
IPOPT as a plain-run ``UpdateClass`` entry, through casadi (ADR 0003).

The registry entry ``"ipopt"`` runs IPOPT (Wächter & Biegler), a
primal-dual interior-point method with a filter line search, on the
loss. By default it uses IPOPT's own limited-memory quasi-Newton
Hessian. With ``hessian="exact"`` it uses ``jax.hessian`` of the loss
instead, each Hessian one recorded, billed evaluation (ADR 0007).

IPOPT owns its loop and calls the objective itself, as scipy does, so
the entry reuses the scipy host-callback machinery unchanged (ADR 0002):
the whole run is one :func:`jax.pure_callback`, one evaluation (value
and gradient together) is one history entry billed one, the run lasts
its whole budget, failure ends it at the best point, and a ``stop_fn``
raises. IPOPT reaches us through casadi, whose wheels bundle it with
the MUMPS linear solver. casadi is imported only when an ``"ipopt"`` run
starts, so importing this package does not load it.

Three things are specific to IPOPT:

* **It asks for the value and the gradient separately**, usually at the
  same point. Each new point is evaluated once, value and gradient
  together, and the most recent one is cached, so the second request
  is free.
* **It cannot be aborted from an evaluation.** When the budget is spent
  or the driver fails, every further request returns NaN, which IPOPT
  treats as an evaluation error, and the iteration callback stops it at
  the end of the iteration. The error is re-raised once IPOPT returns,
  and :func:`~l2co_optimizers._src.scipy_implementations._drive` handles
  it as it would from scipy.
* **The box is honoured exactly.** ``bound_relax_factor`` is zero, so
  IPOPT never evaluates outside the box, and it starts strictly inside
  it (``bound_push``).
* **An exact Hessian is handed to casadi whole.** A callback passed as
  ``nlpsol``'s ``hess_lag`` option returns the upper triangle of
  ``lam_f * H``, with ``H`` from the driver (:meth:`_Driver.hessian`),
  once per IPOPT iteration. Letting casadi derive that function from the
  gradient callback instead costs it ~20 s to build at d = 512 (ADR 0007).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
from typing import Any

# Third-party
import numpy as np
from jax.tree_util import Partial
from jaxtyping import PyTree

# Local
from l2co_optimizers._src.core.state_transfer import FAMILY_GRADIENT
from l2co_optimizers._src.core.typing import LossFunction, StopFunction
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.scipy_implementations import (
    _UNREACHABLE,
    _Driver,
    _scipy_update,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: The registry names this module provides, normalized.
IPOPT_OPTIMIZERS: frozenset[str] = frozenset({"ipopt"})

#: IPOPT's convergence tolerance, as tight as it accepts (it must be
#: positive). With ``acceptable_iter = 0`` disabling the "acceptable
#: level" heuristic, IPOPT ends a run only when it cannot progress.
_IPOPT_TOL = 1e-300

#                                                        Host-side solver
# =============================================================================


def _ipopt_minimize(
    driver: _Driver,
    x0: np.ndarray,
    bounds: Any,
    *,
    options: dict[str, Any],
    exact_hessian: bool = False,
) -> np.ndarray:
    """Run IPOPT to its own end on the driver's objective.

    Follows the ``Minimizer`` contract of ``scipy_implementations``:
    returns IPOPT's final point, and raises what the driver raised (the
    budget spent, a non-finite point) once IPOPT has returned. With
    ``exact_hessian`` IPOPT gets the driver's exact Hessian.
    """
    import casadi as ca

    n = x0.size
    failure: list[Exception] = []
    latest: dict[bytes, tuple[float, np.ndarray]] = {}

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        if failure:
            return np.nan, np.full(n, np.nan)
        key = x.tobytes()
        if key not in latest:
            try:
                out = driver.value_and_grad(x)
            except Exception as error:  # noqa: BLE001 -- re-raised below
                failure.append(error)
                return np.nan, np.full(n, np.nan)
            latest.clear()
            latest[key] = out
        return latest[key]

    def hessian(x: np.ndarray) -> np.ndarray:
        if failure:
            return np.full((n, n), np.nan)
        try:
            return driver.hessian(x)
        except Exception as error:  # noqa: BLE001 -- re-raised below
            failure.append(error)
            return np.full((n, n), np.nan)

    # casadi frees a Python callback whose wrapper is garbage-collected,
    # even while a solver still calls it; hold every one until IPOPT
    # returns.
    alive: list[ca.Callback] = []

    class _LagrangianHessian(ca.Callback):
        """casadi's ``nlp_hess_l``: the upper triangle of ``lam_f * H``.

        Without constraints the Lagrangian is ``lam_f * f``. The inputs
        are casadi's: ``x``, the (empty) parameters, ``lam_f`` and the
        (empty) constraint multipliers.
        """

        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("loss_hessian", {})

        def get_n_in(self):
            return 4

        def get_n_out(self):
            return 1

        def get_name_in(self, i):
            return ("x", "p", "lam_f", "lam_g")[i]

        def get_name_out(self, i):
            return "triu_hess_gamma_x_x"

        def get_sparsity_in(self, i):
            return (
                ca.Sparsity.dense(n, 1),
                ca.Sparsity(0, 0),
                ca.Sparsity.scalar(),
                ca.Sparsity(0, 1),
            )[i]

        def get_sparsity_out(self, i):
            return ca.Sparsity.upper(n)

        def eval(self, arg):
            h = float(arg[2]) * hessian(arg[0].full().ravel())
            return [ca.triu(ca.DM(h))]

    class _Gradient(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("loss_gradient", {})

        def get_n_in(self):
            return 2  # x, and the nominal loss output

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1) if i == 0 else ca.Sparsity.scalar()

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, n)

        def eval(self, arg):
            return [ca.DM(value_and_grad(arg[0].full().ravel())[1]).T]

    class _Loss(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("loss", {})

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.scalar()

        def eval(self, arg):
            return [value_and_grad(arg[0].full().ravel())[0]]

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            gradient = _Gradient()
            alive.append(gradient)
            return gradient

    class _Stop(ca.Callback):
        """Called once per IPOPT iteration; nonzero stops IPOPT."""

        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("stop", {})

        def get_n_in(self):
            return ca.nlpsol_n_out()

        def get_n_out(self):
            return 1

        def get_name_in(self, i):
            return ca.nlpsol_out(i)

        def get_name_out(self, i):
            return "stop"

        def get_sparsity_in(self, i):
            name = ca.nlpsol_out(i)
            if name == "f":
                return ca.Sparsity.scalar()
            if name in ("x", "lam_x"):
                return ca.Sparsity.dense(n)
            return ca.Sparsity(0, 0)  # no constraints, no parameters

        def eval(self, arg):
            return [1 if failure or driver.stalled() else 0]

    loss, stop = _Loss(), _Stop()
    alive += [loss, stop]
    second_order = {}
    if exact_hessian:
        second_order["hess_lag"] = _LagrangianHessian()
        alive.append(second_order["hess_lag"])
    x = ca.MX.sym("x", n)
    solver = ca.nlpsol(
        "ipopt",
        "ipopt",
        {"x": x, "f": loss(x)},
        {
            "iteration_callback": stop,
            "error_on_fail": False,
            # A NaN loss -- the method's to handle -- or the NaN returned
            # after a failure would otherwise print a warning each time.
            "show_eval_warnings": False,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.hessian_approximation": (
                "exact" if exact_hessian else "limited-memory"
            ),
            "ipopt.tol": _IPOPT_TOL,
            "ipopt.acceptable_iter": 0,
            "ipopt.max_iter": _UNREACHABLE,
            "ipopt.bound_relax_factor": 0.0,
            **{f"ipopt.{k}": v for k, v in options.items()},
            **second_order,
        },
    )
    if bounds is None:
        lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
    else:
        lo, hi = bounds.lb, bounds.ub
    result = solver(x0=x0, lbx=lo, ubx=hi)
    if failure:
        raise failure[0]
    return result["x"].full().ravel()


#                                                                Factory
# =============================================================================


def ipopt_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    hessian: str = "limited-memory",
    limited_memory_max_history: int | None = None,
    mu_init: float = 0.1,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="ipopt"``.

    IPOPT, an interior-point method with a filter line search. By
    default it uses its limited-memory quasi-Newton Hessian: without a
    box, the barrier is inactive and it is a line-search L-BFGS method
    with IPOPT's filter and inertia correction. With ``hessian="exact"``
    it is a Newton method on ``jax.hessian`` of the loss (ADR 0007).
    Every value-and-gradient evaluation, and every Hessian, it asks for
    is one evaluation.

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
        Box handed to IPOPT, which never evaluates outside it. ``None``,
        or ``None`` on one side, leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Must be ``None``; see Raises.
    hessian : str, optional
        ``"limited-memory"`` (default) or ``"exact"``: ``jax.hessian``
        of the loss, taken on the sample the point was evaluated with.
    limited_memory_max_history : int or None, optional
        Correction pairs kept by the quasi-Newton Hessian; ``None``
        means IPOPT's default ``6``. Limited-memory only.
    mu_init : float, optional
        Initial barrier parameter; IPOPT's default ``0.1``. Used only
        with a box.

    Returns
    -------
    UpdateClass
        A ``ScipyUpdateClass``: IPOPT runs on the scipy entries'
        host-callback driver.

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``, ``hessian`` is neither
        ``"limited-memory"`` nor ``"exact"``, or
        ``limited_memory_max_history`` is given with ``"exact"``.
    """
    if hessian not in ("limited-memory", "exact"):
        raise ValueError(
            f"Unknown ipopt hessian {hessian!r}; expected 'limited-memory' "
            "or 'exact'."
        )
    exact = hessian == "exact"
    if exact and limited_memory_max_history is not None:
        raise ValueError(
            "limited_memory_max_history configures IPOPT's limited-memory "
            "Hessian; hessian='exact' does not use it."
        )
    options: dict[str, Any] = {"mu_init": mu_init}
    if not exact:
        options["limited_memory_max_history"] = (
            6
            if limited_memory_max_history is None
            else limited_memory_max_history
        )
    return _scipy_update(
        functools.partial(
            _ipopt_minimize, options=options, exact_hessian=exact
        ),
        name="ipopt",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
        with_hessian=exact,
    )


ipopt_mapping = {"ipopt": Partial(ipopt_update)}
