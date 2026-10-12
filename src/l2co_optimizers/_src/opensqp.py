"""
modOpt's OpenSQP as a plain-run ``UpdateClass`` entry (ADR 0010).

The registry entry ``"opensqp"`` runs OpenSQP (Joshy & Hwang), the
sequential quadratic programming (SQP) solver developed inside modOpt.
Each step solves a quadratic model of the problem with a damped BFGS
Hessian, then searches along the result on an augmented-Lagrangian
merit function (a Wolfe line search, with an l1-penalty backtracking
search as a fallback). Bounds enter the quadratic sub-problem as
inequality rows, and every step is clipped to the box, so OpenSQP never
evaluates outside it.

OpenSQP owns its loop and calls the objective itself, as scipy does, so
the entry reuses the scipy host-callback machinery unchanged (ADR 0002):
the whole run is one :func:`jax.pure_callback`, one evaluation (value
and gradient together) is one history entry billed one, the run lasts
its whole budget, failure ends it at the best point, and a ``stop_fn``
raises.

modOpt is not a dependency of this package (ADR 0010). It is on GitHub
only, and its name on PyPI belongs to an unrelated package. Install it
with the ``opensqp`` dependency group (``uv sync --group opensqp``),
which pins it to one commit. The factory imports it, and a missing or
wrong ``modopt`` fails there, before any run starts.

Four things are specific to OpenSQP:

* **It asks for the value and the gradient separately**, at the same
  point. Each new point is evaluated once, value and gradient together,
  and the last :data:`_RECENT_POINTS` are kept, so asking again is free.
* **It can iterate without evaluating.** Its stopping tolerances are
  zero, so at an optimum it can keep taking steps that do not move
  ``x``, or that cycle through the same few points a few units in the
  last place apart. Those points are kept, so the cycle evaluates
  nothing new, and the driver's stall guard ends the run at its current
  point, as for trust-constr (ADR 0003).
* **No wall-clock limit.** When quadprog fails on a quadratic
  sub-problem, OpenSQP solves a relaxed one with HiGHS under a time
  limit, 5 s by default. Here the limit is infinite, so a step never
  depends on how busy the machine is.
* **It writes nothing.** modOpt's output files are turned off.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
import warnings
from collections import OrderedDict
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
OPENSQP_OPTIMIZERS: frozenset[str] = frozenset({"opensqp"})

#: OpenSQP's optimality and feasibility tolerances. At zero it reports
#: convergence only at an exactly stationary point, so a run ends when
#: OpenSQP cannot progress (ADRs 0002, 0010).
_OPENSQP_OPT_TOL = 0.0
_OPENSQP_FEAS_TOL = 0.0

#: Wall-clock limit, in seconds, on HiGHS's solve of the relaxed
#: sub-problem OpenSQP falls back to when quadprog fails. modOpt's
#: default is 5 s; a limit that can bind makes a run depend on the load
#: of the machine it runs on (ADR 0010).
_OPENSQP_AQP_TIME_LIMIT = float("inf")

#: Points whose value and gradient are kept, so that OpenSQP asking for
#: one again evaluates nothing. One iteration evaluates at most about 22
#: points: 10 in the Wolfe line search, 11 in the l1-penalty fallback,
#: and the new iterate. At a box optimum OpenSQP can repeat one
#: iteration's points forever (9 of them at d = 256); kept, they leave the
#: stall guard to end the run (ADR 0010). 64 covers a cycle of two
#: iterations.
_RECENT_POINTS = 64

#: How to install the modOpt this entry was written against.
_INSTALL_HINT = (
    "Install LSDOlab's modOpt from GitHub with this package's opensqp "
    "dependency group (uv sync --group opensqp), or with: pip install "
    "'modopt @ git+https://github.com/LSDOlab/modopt.git' qpsolvers "
    "quadprog highspy. The 'modopt' package on PyPI is an unrelated "
    "library."
)

#                                                        Host-side solver
# =============================================================================


@functools.cache
def _modopt() -> tuple[type, type]:
    """modOpt's ``OpenSQP`` and ``ProblemLite``, and the QP solvers.

    Raises
    ------
    ImportError
        If LSDOlab's modOpt or a QP solver OpenSQP needs is missing.
    """
    try:
        # Import-time noise from modules OpenSQP does not use (invalid
        # escapes, deprecated scipy paths), and "matplotlib not found":
        # modOpt asks for TkAgg, which fails once pyplot is loaded, as
        # f3dasm loads it, and leaves the backend as it was (ADR 0010).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from modopt import OpenSQP, ProblemLite
        import highspy  # noqa: F401
        import qpsolvers  # noqa: F401
        import quadprog  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"optimizer='opensqp' needs modOpt's OpenSQP: {error}. "
            f"{_INSTALL_HINT}"
        ) from error
    return OpenSQP, ProblemLite


class _Stalled(Exception):
    """Raised from OpenSQP's iteration once it stops evaluating."""

    def __init__(self, x: np.ndarray):
        super().__init__()
        self.x = x


def _opensqp_minimize(
    driver: _Driver, x0: np.ndarray, bounds: Any
) -> np.ndarray:
    """Run OpenSQP to its own end on the driver's objective.

    Follows the ``Minimizer`` contract of ``scipy_implementations``:
    returns OpenSQP's final point, and lets what the driver raised (the
    budget spent, a non-finite point) propagate. modOpt catches no
    exception around an evaluation.
    """
    OpenSQP, ProblemLite = _modopt()

    class _GuardedOpenSQP(OpenSQP):
        """OpenSQP, stopped once it iterates without evaluating."""

        def update_outputs(self, **outputs):
            super().update_outputs(**outputs)
            if driver.stalled():
                raise _Stalled(np.array(outputs["x"], dtype=np.float64))

    recent: OrderedDict[bytes, tuple[float, np.ndarray]] = OrderedDict()

    def value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        key = x.tobytes()
        if key not in recent:
            recent[key] = driver.value_and_grad(x)
            while len(recent) > _RECENT_POINTS:
                recent.popitem(last=False)
        return recent[key]

    if bounds is None:
        lo, hi = None, None
    else:
        lo, hi = np.asarray(bounds.lb), np.asarray(bounds.ub)
    with warnings.catch_warnings():
        # "obj_hess not provided": OpenSQP never asks for it.
        warnings.simplefilter("ignore", UserWarning)
        problem = ProblemLite(
            x0,
            obj=lambda x: value_and_grad(x)[0],
            grad=lambda x: value_and_grad(x)[1],
            xl=lo,
            xu=hi,
        )
    solver = _GuardedOpenSQP(
        problem,
        turn_off_outputs=True,
        maxiter=_UNREACHABLE,
        opt_tol=_OPENSQP_OPT_TOL,
        feas_tol=_OPENSQP_FEAS_TOL,
        aqp_time_limit=_OPENSQP_AQP_TIME_LIMIT,
    )
    with warnings.catch_warnings():
        # Repeated on every step once OpenSQP stops moving: qpsolvers
        # converting the relaxed sub-problem for HiGHS, and scipy's BFGS
        # skipping an update for a zero step.
        warnings.filterwarnings("ignore", "Converted ", UserWarning)
        warnings.filterwarnings("ignore", "delta_grad == 0.0", UserWarning)
        try:
            return np.asarray(solver.solve()["x"], dtype=np.float64)
        except _Stalled as stalled:
            return stalled.x


#                                                                Factory
# =============================================================================


def opensqp_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="opensqp"``.

    modOpt's OpenSQP: SQP with a damped BFGS Hessian and a line search
    on an augmented-Lagrangian merit function. Without a box, each step
    is a quasi-Newton step from a dense quadratic program; with one, the
    bounds are inequality rows of that program. It takes inequality and
    equality constraints too; a run passes it only the box (l2co ADR
    0020). It takes no hyperparameters. Its dense quadratic programs
    make the cost per evaluation grow steeply with the dimensionality,
    from a few hundred dimensions with a box and about a thousand
    without (ADR 0010).

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
        Box handed to OpenSQP, which never evaluates outside it.
        ``None``, or ``None`` on one side, leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Must be ``None``; see Raises.

    Returns
    -------
    UpdateClass
        A ``ScipyUpdateClass``: OpenSQP runs on the scipy entries'
        host-callback driver.

    Raises
    ------
    ImportError
        If modOpt, or a QP solver OpenSQP needs, is not installed.
    ValueError
        If ``stop_fn`` is not ``None``.
    """
    _modopt()
    return _scipy_update(
        _opensqp_minimize,
        name="opensqp",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


opensqp_mapping = {"opensqp": Partial(opensqp_update)}
