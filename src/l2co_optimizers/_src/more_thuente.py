"""
The Moré--Thuente line search, one evaluation per call (ADR 0005).

A JAX port of MINPACK-2's ``dcsrch`` and ``dcstep`` as scipy ships them
in ``scipy/optimize/_dcsrch.py`` (scipy 1.18): the line search behind
``scipy.optimize.minimize(method="BFGS")``. It finds a step ``stp``
along a descent direction that satisfies the strong Wolfe conditions

    phi(stp) <= phi(0) + ftol * stp * phi'(0)
    |phi'(stp)| <= gtol * |phi'(0)|

where ``phi`` is the loss along the direction and ``phi'`` its slope.

Like the Fortran original, it is a reverse-communication state machine:
:func:`start` takes the loss and slope at the current point and the
first trial step, and :func:`iterate` takes the loss and slope at the
trial step and either declares the search over or names the next trial.
The caller makes every evaluation. That is what lets a solver drive the
search from inside a JAX scan, one evaluation per step.

Every branch of the original is computed and the right one selected with
``jnp.where``, so the port traces once and runs under ``jit`` and
``vmap``. In float64 it reproduces scipy's sequence of trial steps;
``tests/test_more_thuente.py`` checks it against scipy's own class.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, ScalarLike

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Evaluate the loss and slope at the returned step, then call
#: :func:`iterate` again.
TASK_FG = 0
#: The step satisfies the strong Wolfe conditions.
TASK_CONVERGENCE = 1
#: The search cannot make progress (rounding errors, the ``xtol`` test,
#: or a step pinned at ``stpmin`` or ``stpmax``). scipy reports these as
#: ``WARNING`` and treats them as a failed search.
TASK_WARNING = 2
#: The search was started with an invalid step or a non-descent slope.
TASK_ERROR = 3

_P5 = 0.5
_P66 = 0.66
_XTRAPL = 1.1
_XTRAPU = 4.0


class MoreThuenteState(eqx.Module):
    """What ``dcsrch`` keeps between calls.

    The names are the Fortran original's. ``stx`` is the step with the
    lowest (auxiliary) loss so far, ``sty`` the other end of the interval
    of uncertainty, and ``f*`` / ``g*`` the loss and slope at each.

    Attributes
    ----------
    brackt : Bool[Array, ""]
        Whether a minimiser has been bracketed.
    stage : Int[Array, ""]
        1 until a step with sufficient decrease and a non-negative slope
        is found, 2 after; stage 1 works on the auxiliary function
        ``phi(stp) - ftol * stp * phi'(0)``.
    finit, ginit : Float[Array, ""]
        Loss and slope at step zero.
    gtest : Float[Array, ""]
        ``ftol * ginit``, the sufficient-decrease slope.
    width, width1 : Float[Array, ""]
        The interval width now and before, to force bisection when it
        shrinks too slowly.
    stx, fx, gx : Float[Array, ""]
        Best step so far, with its loss and slope.
    sty, fy, gy : Float[Array, ""]
        The interval's other endpoint, with its loss and slope.
    stmin, stmax : Float[Array, ""]
        Bounds on the next trial step.
    """

    brackt: Bool[Array, ""]
    stage: Int[Array, ""]
    finit: Float[Array, ""]
    ginit: Float[Array, ""]
    gtest: Float[Array, ""]
    width: Float[Array, ""]
    width1: Float[Array, ""]
    stx: Float[Array, ""]
    fx: Float[Array, ""]
    gx: Float[Array, ""]
    sty: Float[Array, ""]
    fy: Float[Array, ""]
    gy: Float[Array, ""]
    stmin: Float[Array, ""]
    stmax: Float[Array, ""]


def start(
    stp: ScalarLike,
    f: ScalarLike,
    g: ScalarLike,
    *,
    ftol: float,
    stpmin: ScalarLike,
    stpmax: ScalarLike,
) -> tuple[MoreThuenteState, Int[Array, ""]]:
    """Begin a search from the loss ``f`` and slope ``g`` at step zero.

    Parameters
    ----------
    stp : ScalarLike
        The first trial step.
    f, g : ScalarLike
        Loss and slope at step zero.
    ftol : float
        Sufficient-decrease constant, in ``(0, 1)``.
    stpmin, stpmax : ScalarLike
        Bounds on every trial step.

    Returns
    -------
    tuple[MoreThuenteState, Int[Array, ""]]
        The state, and :data:`TASK_FG` (evaluate at ``stp``) or
        :data:`TASK_ERROR` when ``stp`` is out of bounds or ``g`` is not
        a finite negative slope, or ``f`` is not finite. The checks on
        the constants are the caller's.
    """
    f = jnp.asarray(f)
    g = jnp.asarray(g, f.dtype)
    stp = jnp.asarray(stp, f.dtype)
    error = (
        (stp < stpmin)
        | (stp > stpmax)
        | ~(g < 0)
        | ~jnp.isfinite(g)
        | ~jnp.isfinite(f)
    )
    width = jnp.asarray(stpmax - stpmin, f.dtype)
    zero = jnp.zeros((), f.dtype)
    state = MoreThuenteState(
        brackt=jnp.array(False),
        stage=jnp.array(1),
        finit=f,
        ginit=g,
        gtest=ftol * g,
        width=width,
        width1=width / _P5,
        stx=zero,
        fx=f,
        gx=g,
        sty=zero,
        fy=f,
        gy=g,
        stmin=zero,
        stmax=stp + _XTRAPU * stp,
    )
    return state, jnp.where(error, TASK_ERROR, TASK_FG)


def iterate(
    state: MoreThuenteState,
    stp: ScalarLike,
    f: ScalarLike,
    g: ScalarLike,
    *,
    ftol: float,
    gtol: float,
    xtol: float,
    stpmin: ScalarLike,
    stpmax: ScalarLike,
) -> tuple[MoreThuenteState, Float[Array, ""], Int[Array, ""]]:
    """Take the loss ``f`` and slope ``g`` at the trial step ``stp``.

    Parameters
    ----------
    state : MoreThuenteState
        The search so far.
    stp : ScalarLike
        The step that was evaluated.
    f, g : ScalarLike
        Loss and slope at ``stp``.
    ftol, gtol : float
        The strong Wolfe constants, ``0 < ftol < gtol < 1``.
    xtol : float
        Relative width below which the interval is too small to search.
    stpmin, stpmax : ScalarLike
        Bounds on every trial step.

    Returns
    -------
    tuple[MoreThuenteState, Float[Array, ""], Int[Array, ""]]
        The new state, a step, and a task. On :data:`TASK_FG` the step
        is the next trial; on :data:`TASK_CONVERGENCE` or
        :data:`TASK_WARNING` it is ``stp`` itself and the search is over.
    """
    s = state
    f = jnp.asarray(f, s.finit.dtype)
    g = jnp.asarray(g, s.finit.dtype)
    stp = jnp.asarray(stp, s.finit.dtype)

    ftest = s.finit + stp * s.gtest
    stage = jnp.where((s.stage == 1) & (f <= ftest) & (g >= 0), 2, s.stage)

    # Later tests override earlier ones, as in the original.
    warning = (
        (s.brackt & ((stp <= s.stmin) | (stp >= s.stmax)))
        | (s.brackt & (s.stmax - s.stmin <= xtol * s.stmax))
        | ((stp == stpmax) & (f <= ftest) & (g <= s.gtest))
        | ((stp == stpmin) & ((f > ftest) | (g >= s.gtest)))
    )
    convergence = (f <= ftest) & (jnp.abs(g) <= gtol * -s.ginit)
    done = jnp.where(
        convergence,
        TASK_CONVERGENCE,
        jnp.where(warning, TASK_WARNING, TASK_FG),
    )

    # In stage 1, a step with a lower loss than the best but without
    # sufficient decrease takes the auxiliary (``gtest``-shifted) values.
    modified = (stage == 1) & (f <= s.fx) & (f > ftest)
    gtest = s.gtest
    m_step = dcstep(
        s.stx,
        s.fx - s.stx * gtest,
        s.gx - gtest,
        s.sty,
        s.fy - s.sty * gtest,
        s.gy - gtest,
        stp,
        f - stp * gtest,
        g - gtest,
        s.brackt,
        s.stmin,
        s.stmax,
    )
    m_stx, m_fxm, m_gxm, m_sty, m_fym, m_gym, m_stp, m_brackt = m_step
    m_step = (
        m_stx,
        m_fxm + m_stx * gtest,
        m_gxm + gtest,
        m_sty,
        m_fym + m_sty * gtest,
        m_gym + gtest,
        m_stp,
        m_brackt,
    )
    u_step = dcstep(
        s.stx,
        s.fx,
        s.gx,
        s.sty,
        s.fy,
        s.gy,
        stp,
        f,
        g,
        s.brackt,
        s.stmin,
        s.stmax,
    )
    stx, fx, gx, sty, fy, gy, new_stp, brackt = (
        jnp.where(modified, m, u) for m, u in zip(m_step, u_step, strict=True)
    )

    # Force a bisection when the interval shrinks too slowly.
    bisect = brackt & (jnp.abs(sty - stx) >= _P66 * s.width1)
    new_stp = jnp.where(bisect, stx + _P5 * (sty - stx), new_stp)
    width1 = jnp.where(brackt, s.width, s.width1)
    width = jnp.where(brackt, jnp.abs(sty - stx), s.width)

    stmin = jnp.where(
        brackt,
        jnp.minimum(stx, sty),
        new_stp + _XTRAPL * (new_stp - stx),
    )
    stmax = jnp.where(
        brackt,
        jnp.maximum(stx, sty),
        new_stp + _XTRAPU * (new_stp - stx),
    )

    new_stp = jnp.clip(new_stp, stpmin, stpmax)
    stuck = (brackt & ((new_stp <= stmin) | (new_stp >= stmax))) | (
        brackt & (stmax - stmin <= xtol * stmax)
    )
    new_stp = jnp.where(stuck, stx, new_stp)

    advanced = MoreThuenteState(
        brackt=brackt,
        stage=stage,
        finit=s.finit,
        ginit=s.ginit,
        gtest=s.gtest,
        width=width,
        width1=width1,
        stx=stx,
        fx=fx,
        gx=gx,
        sty=sty,
        fy=fy,
        gy=gy,
        stmin=stmin,
        stmax=stmax,
    )
    finished = eqx.tree_at(lambda t: t.stage, s, stage)
    searching = done == TASK_FG
    new_state = _tree_where(searching, advanced, finished)
    return new_state, jnp.where(searching, new_stp, stp), done


def dcstep(
    stx: Float[Array, ""],
    fx: Float[Array, ""],
    dx: Float[Array, ""],
    sty: Float[Array, ""],
    fy: Float[Array, ""],
    dy: Float[Array, ""],
    stp: Float[Array, ""],
    fp: Float[Array, ""],
    dp: Float[Array, ""],
    brackt: Bool[Array, ""],
    stpmin: ScalarLike,
    stpmax: ScalarLike,
) -> tuple[Float[Array, ""], ...]:
    """One safeguarded step of the interval of uncertainty.

    ``(stx, fx, dx)`` is the best step so far, ``(sty, fy, dy)`` the
    interval's other endpoint and ``(stp, fp, dp)`` the trial; ``d*``
    are slopes. Returns the updated ``stx, fx, dx, sty, fy, dy``, the
    next trial step and ``brackt``. All four cases of the original are
    computed; where a case does not apply its arithmetic may produce a
    NaN or an infinity, which ``jnp.where`` discards.
    """
    sgnd = jnp.sign(dp) * jnp.sign(dx)

    # Case 1: a higher loss. The minimiser is bracketed.
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
    gamma = s * jnp.sqrt((theta / s) ** 2 - dx / s * (dp / s))
    gamma = jnp.where(stp < stx, -gamma, gamma)
    p = gamma - dx + theta
    q = gamma - dx + gamma + dp
    r = p / q
    stpc = stx + r * (stp - stx)
    stpq = stx + dx / ((fx - fp) / (stp - stx) + dx) / 2.0 * (stp - stx)
    stpf1 = jnp.where(
        jnp.abs(stpc - stx) <= jnp.abs(stpq - stx),
        stpc,
        stpc + (stpq - stpc) / 2.0,
    )

    # Case 2: a lower loss and slopes of opposite sign. Bracketed.
    gamma = s * jnp.sqrt((theta / s) ** 2 - dx / s * (dp / s))
    gamma = jnp.where(stp > stx, -gamma, gamma)
    p = gamma - dp + theta
    q = gamma - dp + gamma + dx
    r = p / q
    stpc = stp + r * (stx - stp)
    stpq2 = stp + dp / (dp - dx) * (stx - stp)
    stpf2 = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq2 - stp), stpc, stpq2)

    # Case 3: a lower loss, slopes of the same sign, the slope shrinking.
    gamma = s * jnp.sqrt(
        jnp.maximum(0.0, (theta / s) ** 2 - dx / s * (dp / s))
    )
    gamma = jnp.where(stp > stx, -gamma, gamma)
    p = gamma - dp + theta
    q = gamma + (dx - dp) + gamma
    r = p / q
    stpc = jnp.where(
        (r < 0) & (gamma != 0),
        stp + r * (stx - stp),
        jnp.where(stp > stx, stpmax, stpmin),
    )
    stpq = stp + dp / (dp - dx) * (stx - stp)
    bracketed = jnp.where(
        jnp.abs(stpc - stp) < jnp.abs(stpq - stp), stpc, stpq
    )
    bracketed = jnp.where(
        stp > stx,
        jnp.minimum(stp + _P66 * (sty - stp), bracketed),
        jnp.maximum(stp + _P66 * (sty - stp), bracketed),
    )
    unbracketed = jnp.clip(
        jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq),
        stpmin,
        stpmax,
    )
    stpf3 = jnp.where(brackt, bracketed, unbracketed)

    # Case 4: a lower loss, slopes of the same sign, the slope not
    # shrinking.
    theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dy)), jnp.abs(dp))
    gamma = s * jnp.sqrt((theta / s) ** 2 - dy / s * (dp / s))
    gamma = jnp.where(stp > sty, -gamma, gamma)
    p = gamma - dp + theta
    q = gamma - dp + gamma + dy
    r = p / q
    stpf4 = jnp.where(
        brackt,
        stp + r * (sty - stp),
        jnp.where(stp > stx, stpmax, stpmin),
    )

    case1 = fp > fx
    case2 = ~case1 & (sgnd < 0)
    case3 = ~case1 & ~case2 & (jnp.abs(dp) < jnp.abs(dx))
    stpf = jnp.where(
        case1,
        stpf1,
        jnp.where(case2, stpf2, jnp.where(case3, stpf3, stpf4)),
    )
    brackt = brackt | case1 | case2

    # Update the interval of uncertainty.
    to_y = case1
    swap = ~case1 & (sgnd < 0)
    new_sty = jnp.where(to_y, stp, jnp.where(swap, stx, sty))
    new_fy = jnp.where(to_y, fp, jnp.where(swap, fx, fy))
    new_dy = jnp.where(to_y, dp, jnp.where(swap, dx, dy))
    new_stx = jnp.where(to_y, stx, stp)
    new_fx = jnp.where(to_y, fx, fp)
    new_dx = jnp.where(to_y, dx, dp)
    return new_stx, new_fx, new_dx, new_sty, new_fy, new_dy, stpf, brackt


def _tree_where(
    pred: Bool[Array, ""], true: MoreThuenteState, false: MoreThuenteState
) -> MoreThuenteState:
    """Select between two states, leaf by leaf."""
    return jax.tree.map(lambda t, f: jnp.where(pred, t, f), true, false)
