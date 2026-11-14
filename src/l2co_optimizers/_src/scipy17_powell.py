"""
scipy 1.7.3's bounded Powell, vendored for ``"mlsl"``.

The MLSL reference (``dynas/mlsl.py``, Zenodo 10.5281/zenodo.6458266) gives
its local searches a shared pool of evaluations that runs out early, after
which it calls Powell with ``maxfev = 0``. scipy from 1.8.0 on refuses that
call before evaluating anything (``_MaxFuncCallError``); scipy up to 1.7
evaluated the start point and ran one full sweep of line searches before it
first compared the call count with ``maxfev``. The Zenodo runs could only
have finished under the old behaviour (the reference has no handler for the
error), so ``"mlsl"`` uses this copy of the old Powell rather than the
scipy the repo locks. See the ``mlsl`` module docstring, deviation 6.

Source: ``scipy/optimize/optimize.py`` at tag v1.7.3
(https://github.com/scipy/scipy/blob/v1.7.3/scipy/optimize/optimize.py):
``_minimize_powell``, ``_linesearch_powell``, ``_line_for_search`` and
``_minimize_scalar_bounded``. Edits, none of which changes which points are
evaluated or in what order:

1. ``_minimize_powell`` keeps only the arguments ``"mlsl"`` passes (``func``,
   ``x0``, ``bounds``, ``xtol``, ``ftol``, ``maxiter``, ``maxfev``); the
   ``callback``, ``disp``, ``direc``, ``return_all`` and ``args`` paths and
   the out-of-bounds start warning are removed.
2. ``_linesearch_powell`` keeps only the branch for a direction bounded on
   both sides, the only one a finite box reaches; the unbounded and
   one-sided branches raise instead.
3. ``_minimize_scalar_bounded`` drops ``disp`` printing and its argument
   validation (the caller always passes a valid finite interval).
4. Status messages are inlined.

Copyright (c) 2001-2002 Enthought, Inc. 2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from math import sqrt

# Third-party
import numpy as np
from scipy.optimize import Bounds, OptimizeResult

# =============================================================================


def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """Bounds ``(lmin, lmax)`` on ``l`` keeping ``x0 + alpha * l`` in box."""
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    (nonzero,) = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _minimize_scalar_bounded(func, bounds, xatol=1e-5, maxiter=500):
    """Bounded Brent (fminbound), as in scipy 1.7.3."""
    maxfun = maxiter
    x1, x2 = bounds
    flag = 0

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x)
    num = 1
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while np.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if (
                (np.abs(p) < np.abs(0.5 * q * r))
                and (p > q * (a - xf))
                and (p < q * (b - xf))
            ):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:  # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x)
        num += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    return OptimizeResult(
        fun=fx, status=flag, success=(flag == 0), x=xf, nfev=num, nit=num
    )


def _linesearch_powell(
    func, p, xi, tol=1e-3, lower_bound=None, upper_bound=None, fval=None
):
    """Line search along ``xi`` from ``p``, bounded on both sides."""

    def myfunc(alpha):
        return func(p + alpha * xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return (fval, p, xi) if fval is not None else (func(p), p, xi)
    bound = _line_for_search(p, xi, lower_bound, upper_bound)
    if np.isneginf(bound[0]) or np.isposinf(bound[1]):
        raise ValueError(
            "the vendored Powell supports only a finite box (edit 2)"
        )
    # we can use a bounded scalar minimization
    res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
    xi = res.x * xi
    return np.squeeze(res.fun), p + xi, xi


def minimize_powell(
    func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Bounds,
    xtol: float = 1e-4,
    ftol: float = 1e-4,
    maxiter: float | None = None,
    maxfev: float | None = None,
) -> OptimizeResult:
    """``scipy.optimize.minimize(method="Powell")`` as in scipy 1.7.3.

    The difference that matters from current scipy: ``maxfev`` is checked
    only after each full sweep of line searches, so a call with
    ``maxfev = 0`` still evaluates ``x0`` and runs one sweep (and a call
    with a small ``maxfev`` may exceed it by up to a sweep).

    Parameters
    ----------
    func
        Scalar objective of a 1-D array.
    x0
        Start point.
    bounds
        A finite box.
    xtol, ftol, maxiter, maxfev
        As in scipy's Powell.

    Returns
    -------
    OptimizeResult
        With ``x``, ``fun``, ``nfev``, ``nit``, ``status`` and ``success``.
    """
    maxfun = maxfev
    fcalls = [0]

    def counted(x):
        fcalls[0] += 1
        return func(np.copy(x))

    x = np.asarray(x0).flatten()
    N = len(x)
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    direc = np.eye(N, dtype=float)
    lower_bound, upper_bound = bounds.lb, bounds.ub

    fval = np.squeeze(counted(x))
    x1 = x.copy()
    iter = 0
    ilist = list(range(N))
    while True:
        fx = fval
        bigind = 0
        delta = 0.0
        for i in ilist:
            direc1 = direc[i]
            fx2 = fval
            fval, x, direc1 = _linesearch_powell(
                counted,
                x,
                direc1,
                tol=xtol * 100,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                fval=fval,
            )
            if (fx2 - fval) > delta:
                delta = fx2 - fval
                bigind = i
        iter += 1
        bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
        if 2.0 * (fx - fval) <= bnd:
            break
        if fcalls[0] >= maxfun:
            break
        if iter >= maxiter:
            break
        if np.isnan(fx) and np.isnan(fval):
            # Ended up in a nan-region: bail out
            break

        # Construct the extrapolated point
        direc1 = x - x1
        x2 = 2 * x - x1
        x1 = x.copy()
        fx2 = np.squeeze(counted(x2))

        if fx > fx2:
            t = 2.0 * (fx + fx2 - 2.0 * fval)
            temp = fx - fval - delta
            t *= temp * temp
            temp = fx - fx2
            t -= delta * temp * temp
            if t < 0.0:
                fval, x, direc1 = _linesearch_powell(
                    counted,
                    x,
                    direc1,
                    tol=xtol * 100,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    fval=fval,
                )
                if np.any(direc1):
                    direc[bigind] = direc[-1]
                    direc[-1] = direc1

    warnflag = 0
    if np.any(lower_bound > x) or np.any(x > upper_bound):
        warnflag = 4
    elif fcalls[0] >= maxfun:
        warnflag = 1
    elif iter >= maxiter:
        warnflag = 2
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3

    return OptimizeResult(
        fun=fval,
        direc=direc,
        nit=iter,
        nfev=fcalls[0],
        status=warnflag,
        success=(warnflag == 0),
        x=x,
    )
