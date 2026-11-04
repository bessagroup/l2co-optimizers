"""ADR 0002's whole-run L-BFGS-B, kept as the reference for the stepped one.

Until ADR 0011 stepped it, ``"lbfgsb"`` ran its whole budget as one
``scipy.optimize.minimize`` call inside one host callback, on the driver
the other scipy entries still use
(:func:`~l2co_optimizers._src.scipy_implementations._scipy_update`). The
databank's ``lbfgsb`` cells come from that driver. This module rebuilds
it, exactly as it shipped, so the tests can hold the stepped entry to it
bitwise. It is a validation reference only and is not shipped.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import scipy.optimize as so

# Local
from l2co_optimizers import FAMILY_GRADIENT
from l2co_optimizers._src.scipy_implementations import (
    _LBFGSB_FTOL,
    _LBFGSB_GTOL,
    _UNREACHABLE,
    _scipy_update,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def whole_run_lbfgsb_minimize(maxcor=10, maxls=20, record=None):
    """The ``minimize`` closure the whole-run ``"lbfgsb"`` handed its driver.

    With ``record`` (a list), every ``(x, f, g)`` scipy is handed is
    appended to it, in order: the evaluations a stepped L-BFGS-B must
    replay into the same points.
    """

    def minimize(driver, x0, bounds):
        fun = driver.value_and_grad
        if record is not None:

            def fun(x):
                f, g = driver.value_and_grad(x)
                record.append((x.copy(), f, g.copy()))
                return f, g

        return so.minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(
                maxcor=maxcor,
                maxls=maxls,
                ftol=_LBFGSB_FTOL,
                gtol=_LBFGSB_GTOL,
                maxfun=_UNREACHABLE,
                maxiter=_UNREACHABLE,
            ),
        ).x

    return minimize


def whole_run_lbfgsb_update(
    *,
    model,
    loss_fn,
    pass_rng,
    opt_hash,
    bounded=(None, None),
    maxcor=10,
    maxls=20,
    record=None,
):
    """The whole-run ``"lbfgsb"`` ``UpdateClass``, as ADR 0006 built it."""
    return _scipy_update(
        whole_run_lbfgsb_minimize(maxcor, maxls, record),
        name="lbfgsb",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=None,
    )
