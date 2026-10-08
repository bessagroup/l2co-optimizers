"""The Moré--Thuente port reproduces scipy's ``DCSRCH`` (ADR 0005).

Both are driven with the same loss and slope values along a line, and
must name the same trial steps and end the same way: converged, warned
or refused at the start. Float64 throughout, as scipy is.
"""

#                                                                       Modules
# =============================================================================

# Standard
import functools

# Third-party
import equinox as eqx
import jax
import numpy as np
import pytest

# Local
from l2co_optimizers._src import more_thuente as mt

scipy_dcsrch = pytest.importorskip("scipy.optimize._dcsrch")

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: The constants ``scipy.optimize.minimize(method="BFGS")`` searches with.
BFGS = dict(ftol=1e-4, gtol=0.9, xtol=1e-14, stpmin=1e-100, stpmax=1e100)

TASKS = {
    b"FG": mt.TASK_FG,
    b"CONV": mt.TASK_CONVERGENCE,
    b"WARN": mt.TASK_WARNING,
    b"ERRO": mt.TASK_ERROR,
}


def _scipy_trace(phi, dphi, stp, c, maxiter=100):
    """Every ``(step, task)`` scipy's ``DCSRCH`` names, in order."""
    search = scipy_dcsrch.DCSRCH(
        phi, dphi, c["ftol"], c["gtol"], c["xtol"], c["stpmin"], c["stpmax"]
    )
    stp, _, _, task = search._iterate(stp, phi(0.0), dphi(0.0), b"START")
    trace = [(stp, TASKS[task[:4] if task[:2] != b"FG" else b"FG"])]
    for _ in range(maxiter):
        if task[:2] != b"FG":
            break
        stp, _, _, task = search._iterate(stp, phi(stp), dphi(stp), task)
        trace.append((stp, TASKS[task[:4] if task[:2] != b"FG" else b"FG"]))
    return trace


@functools.cache
def _jitted_iterate(constants):
    """One compiled :func:`more_thuente.iterate` per set of constants."""
    c = dict(constants)
    return eqx.filter_jit(
        lambda state, stp, f, g: mt.iterate(state, stp, f, g, **c)
    )


def _port_trace(phi, dphi, stp, c, maxiter=100):
    """The same for :mod:`more_thuente`, one jitted call per evaluation."""
    iterate = _jitted_iterate(tuple(sorted(c.items())))
    state, task = mt.start(
        stp,
        phi(0.0),
        dphi(0.0),
        ftol=c["ftol"],
        stpmin=c["stpmin"],
        stpmax=c["stpmax"],
    )
    trace = [(stp, int(task))]
    for _ in range(maxiter):
        if trace[-1][1] != mt.TASK_FG:
            break
        # 0-d arrays, not floats: filter_jit would treat a float as static
        # and compile once per value.
        state, stp, task = iterate(
            state, *(np.asarray(v) for v in (stp, phi(stp), dphi(stp)))
        )
        stp = float(stp)
        trace.append((stp, int(task)))
    return trace


def _assert_same(phi, dphi, stp, c=BFGS):
    with jax.enable_x64(True):
        port = _port_trace(phi, dphi, stp, c)
    reference = _scipy_trace(phi, dphi, stp, c)
    assert [t for _, t in port] == [t for _, t in reference]
    np.testing.assert_allclose(
        [s for s, _ in port], [s for s, _ in reference], rtol=1e-12, atol=0
    )
    return reference


def _rosenbrock_line(seed, dim, scale):
    """Rosenbrock along a descent direction from a random point."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, dim)
    a = rng.normal(size=(dim, dim))
    hinv = a @ a.T / dim + 0.1 * np.eye(dim)  # a random SPD metric

    def f(z):
        return np.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2)

    def grad(z):
        g = np.zeros_like(z)
        g[:-1] = -400.0 * z[:-1] * (z[1:] - z[:-1] ** 2) - 2 * (1 - z[:-1])
        g[1:] += 200.0 * (z[1:] - z[:-1] ** 2)
        return g

    p = -hinv @ grad(x)
    return (
        lambda s: float(f(x + s * p)),
        lambda s: float(grad(x + s * p) @ p),
        scale / np.linalg.norm(p),
    )


# =============================================================================
#                                                          Same trial steps


@pytest.mark.parametrize(
    ("phi", "dphi", "stp"),
    [
        # Overshoots: the minimiser at 0.3 must be bracketed.
        (lambda s: (s - 0.3) ** 2, lambda s: 2 * (s - 0.3), 1.0),
        # Undershoots: steps must be extrapolated out to 20.
        (lambda s: (s - 20.0) ** 2, lambda s: 2 * (s - 20.0), 1.0),
        # A flat minimum.
        (lambda s: (s - 1.0) ** 4, lambda s: 4 * (s - 1.0) ** 3, 3.0),
        # Lower loss than the start but not enough decrease at first
        # (stage 1, the auxiliary function).
        (
            lambda s: -s + 0.6 * s**2 - 0.1 * s**3 + 0.01 * s**4,
            lambda s: -1 + 1.2 * s - 0.3 * s**2 + 0.04 * s**3,
            8.0,
        ),
    ],
    ids=["bracket", "extrapolate", "flat", "auxiliary"],
)
def test_one_dimensional_lines(phi, dphi, stp):
    trace = _assert_same(phi, dphi, stp)
    assert trace[-1][1] == mt.TASK_CONVERGENCE


@pytest.mark.parametrize(
    "scale", [1e-6, 1.0, 1e3], ids=["short", "unit", "long"]
)
@pytest.mark.parametrize("seed", range(12))
def test_rosenbrock_lines(seed, scale):
    """Random points and metrics: every branch of ``dcstep`` gets used."""
    phi, dphi, stp = _rosenbrock_line(seed, 2 + seed % 5, scale)
    _assert_same(phi, dphi, stp)


# =============================================================================
#                                                      Same way of failing


def test_a_step_pinned_at_stpmax_warns():
    """Unbounded below: the step runs into ``stpmax``."""
    c = dict(BFGS, stpmax=10.0)
    trace = _assert_same(lambda s: -s, lambda s: -1.0, 1.0, c)
    assert trace[-1] == (10.0, mt.TASK_WARNING)


@pytest.mark.parametrize(
    ("phi", "dphi"),
    [
        # The loss jumps up while the slope still says downhill.
        (lambda s: 1.0 if s < 0.5 else 2.0, lambda s: -1.0),
        # A flat loss whose slope changes sign: the interval collapses.
        (lambda s: 1.0, lambda s: -1e-3 if s < 0.5 else 1e-3),
    ],
    ids=["rounding", "xtol"],
)
def test_inconsistent_losses_warn(phi, dphi):
    """What rounding errors look like to the search."""
    trace = _assert_same(phi, dphi, 1.0)
    assert trace[-1][1] == mt.TASK_WARNING


@pytest.mark.parametrize(
    ("stp", "g0", "c"),
    [
        (1.0, 0.0, BFGS),  # not a descent direction
        (1.0, 2.0, BFGS),  # uphill
        (1.0, np.nan, BFGS),
        (20.0, -1.0, dict(BFGS, stpmax=10.0)),
    ],
    ids=["flat", "uphill", "nan", "beyond-stpmax"],
)
def test_bad_starts_are_errors(stp, g0, c):
    with jax.enable_x64(True):
        _, task = mt.start(
            stp,
            1.0,
            g0,
            ftol=c["ftol"],
            stpmin=c["stpmin"],
            stpmax=c["stpmax"],
        )
    assert int(task) == mt.TASK_ERROR
    if not np.isnan(g0):  # scipy lets a NaN slope through
        reference = _scipy_trace(lambda s: 1.0, lambda s: g0, stp, c)
        assert reference == [(stp, mt.TASK_ERROR)]


def test_runs_under_vmap():
    """Each line of a batch follows its own search."""
    with jax.enable_x64(True):
        targets = np.array([0.3, 20.0])
        state, _ = jax.vmap(
            lambda t: mt.start(
                1.0, t**2, -2 * t, ftol=1e-4, stpmin=1e-100, stpmax=1e100
            )
        )(targets)
        stp = np.ones(2)
        state, stp, task = jax.vmap(
            lambda s, x, t: mt.iterate(s, x, (x - t) ** 2, 2 * (x - t), **BFGS)
        )(state, stp, targets)
        for target, step in zip(targets, np.asarray(stp), strict=True):
            alone = _port_trace(
                lambda s, t=target: (s - t) ** 2,
                lambda s, t=target: 2 * (s - t),
                1.0,
                BFGS,
            )
            assert step == pytest.approx(alone[1][0], rel=1e-12)
