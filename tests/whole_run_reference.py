"""ADR 0002's whole-run L-BFGS-B and SLSQP, the references for the stepped.

Until ADR 0011 stepped them, ``"lbfgsb"`` and ``"slsqp"`` ran their
whole budget as one ``scipy.optimize.minimize`` call inside one host
callback, on the driver the other scipy entries still use
(:func:`~l2co_optimizers._src.scipy_implementations._scipy_update`). The
databank's ``lbfgsb`` and ``slsqp`` cells come from that driver. This
module rebuilds both entries, exactly as they shipped, so the tests can
hold the stepped ones to them bitwise. It is a validation reference
only and is not shipped.
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
    _SLSQP_FTOL,
    _UNREACHABLE,
    _scipy_update,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def whole_run_lbfgsb_minimize(maxcor=10, maxls=20):
    """The ``minimize`` call the whole-run ``"lbfgsb"`` handed its driver."""

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value_and_grad,
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
):
    """The whole-run ``"lbfgsb"`` ``UpdateClass``, as ADR 0006 built it."""
    return _scipy_update(
        whole_run_lbfgsb_minimize(maxcor, maxls),
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


def whole_run_slsqp_minimize():
    """The ``minimize`` call the whole-run ``"slsqp"`` handed its driver."""

    def minimize(driver, x0, bounds):
        return so.minimize(
            driver.value_and_grad,
            x0,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            options=dict(ftol=_SLSQP_FTOL, maxiter=_UNREACHABLE),
        ).x

    return minimize


def whole_run_slsqp_update(
    *, model, loss_fn, pass_rng, opt_hash, bounded=(None, None)
):
    """The whole-run ``"slsqp"`` ``UpdateClass``, as ADR 0003 built it."""
    return _scipy_update(
        whole_run_slsqp_minimize(),
        name="slsqp",
        family=FAMILY_GRADIENT,
        with_grad=True,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=None,
    )
