"""Unit tests for the OpenSQP registry entry (ADR 0010).

``"opensqp"`` hands a whole run to modOpt's OpenSQP inside one host
callback, reusing the scipy entries' driver (ADR 0002). The properties
checked are those of ``test_ipopt.py``, plus OpenSQP's own:

1. the wrapper is transparent -- its evaluations are the ones a direct
   modOpt run makes, until OpenSQP stops; and OpenSQP's separate value
   and gradient requests at one point are one evaluation;
2. it improves on its starting point;
3. one iteration is one real evaluation, billed one, for exactly
   ``n_iterations``;
4. after OpenSQP stops, or stops evaluating, its final point is
   re-evaluated to the budget;
5. every evaluated point lies in ``bounded``;
6. the settings that make a run last its budget, and keep it off the
   wall clock, reach OpenSQP; it writes no files, and importing modOpt
   leaves matplotlib's backend alone;
7. ``stop_fn`` raises, ``step`` raises, it does not unpack into switching
   parts, and realizations run one at a time.

modOpt is installed from GitHub with the ``opensqp`` dependency group.
Without it this module skips, unless ``L2CO_OPTIMIZERS_REQUIRE_MODOPT``
is set, as CI sets it: there a missing modOpt fails. What the factory
does without modOpt is tested in ``test_opensqp_dependency.py``.
"""

#                                                                       Modules
# =============================================================================

# Standard
import os
import subprocess
import sys

# Third-party
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    FAMILY_GRADIENT,
    PLAIN_RUN_OPTIMIZERS,
    OptimizationStep,
    optimizer_parts,
)
from l2co_optimizers._src import opensqp as opensqp_module
from l2co_optimizers._src.mapping import optimizer_mapping, optimizers
from l2co_optimizers._src.opensqp import (
    _OPENSQP_AQP_TIME_LIMIT,
    _OPENSQP_FEAS_TOL,
    _OPENSQP_OPT_TOL,
    _UNREACHABLE,
    OPENSQP_OPTIMIZERS,
    _modopt,
    opensqp_update,
)
from l2co_optimizers._src.scipy_implementations import ScipyUpdateClass

from .test_scipy import (
    DIM,
    _build,
    _counting,
    _losses,
    _run,
    _wavy,
    _x0,
)
from .toy_problems import quadratic_problem

try:
    OpenSQP, ProblemLite = _modopt()
except ImportError as error:
    if os.environ.get("L2CO_OPTIMIZERS_REQUIRE_MODOPT"):
        raise
    pytest.skip(
        f"modOpt is not installed (uv sync --group opensqp): {error}",
        allow_module_level=True,
    )

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def _spy_on_modopt(monkeypatch):
    """Record every OpenSQP and ProblemLite the entry builds."""
    built = {"solver": [], "problem": []}

    class SpyOpenSQP(OpenSQP):
        def __init__(self, problem, **options):
            built["solver"].append(options)
            super().__init__(problem, **options)

    class SpyProblemLite(ProblemLite):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built["problem"].append(self)

    monkeypatch.setattr(
        opensqp_module, "_modopt", lambda: (SpyOpenSQP, SpyProblemLite)
    )
    return built


def test_registered_under_its_bare_name():
    assert "opensqp" in OPENSQP_OPTIMIZERS
    assert "opensqp" in PLAIN_RUN_OPTIMIZERS
    assert optimizers["opensqp"].func is opensqp_update
    assert optimizer_mapping("OpenSQP") is optimizers["opensqp"]


# =============================================================================
#                                                1. transparent wrapper


def _direct_opensqp(loss_fn, x0, maxiter):
    """The losses a direct modOpt OpenSQP run evaluates, one per new point.

    An independent re-statement of the wrapper: OpenSQP asks for the
    value and the gradient separately, and only the first request at a
    point is recorded.
    """
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    seen, points = [], {}

    def at(x):
        x = np.asarray(x, dtype=np.float64)
        if x.tobytes() not in points:
            v, g = value_and_grad(jnp.asarray(x))
            points[x.tobytes()] = (float(v), np.asarray(g, dtype=np.float64))
            seen.append(float(v))
        return points[x.tobytes()]

    problem = ProblemLite(
        np.asarray(x0, dtype=np.float64),
        obj=lambda x: at(x)[0],
        grad=lambda x: at(x)[1],
    )
    OpenSQP(
        problem,
        turn_off_outputs=True,
        maxiter=maxiter,
        opt_tol=_OPENSQP_OPT_TOL,
        feas_tol=_OPENSQP_FEAS_TOL,
        aqp_time_limit=_OPENSQP_AQP_TIME_LIMIT,
    ).solve()
    return np.asarray(seen)


def test_evaluations_match_a_direct_opensqp_run():
    x0 = _x0()
    direct = _direct_opensqp(_wavy, x0[0], maxiter=300)
    n = 400
    ours = _losses(_run(_build("opensqp"), n, params=x0))
    k = min(len(direct), n)
    assert k > 10, f"OpenSQP stopped after {k} evaluations"
    np.testing.assert_allclose(ours[:k], direct[:k], rtol=1e-6)


def test_value_and_gradient_at_one_point_are_one_evaluation(monkeypatch):
    """OpenSQP asks for the value and the gradient separately at each
    point; each point is billed once, as for IPOPT (ADR 0003)."""
    built = _spy_on_modopt(monkeypatch)
    calls = []
    n = 100
    _run(_build("opensqp", _counting(_wavy, calls)), n)
    jax.effects_barrier()
    problem = built["problem"][0]
    # ProblemLite counts each value request and each gradient request.
    assert problem.nfev + problem.ngev > len(calls) == n
    assert problem.ngev > n // 2


# =============================================================================
#                                                 2. and 3. progress, billing


def test_improves_on_its_starting_point():
    out = _run(_build("opensqp"), 300, params=3.0 + _x0())
    assert float(out[2]) < 0.2 * _losses(out)[0]


def test_each_iteration_is_one_real_evaluation():
    calls = []
    n = 120
    out = _run(_build("opensqp", _counting(_wavy, calls)), n)
    jax.effects_barrier()
    history = out[5]
    assert len(calls) == n
    assert int(history.cursor) == n
    np.testing.assert_array_equal(np.asarray(history.fevals), 1)
    np.testing.assert_array_equal(np.asarray(history.iterations), 1)
    np.testing.assert_array_equal(np.asarray(history.update_step), 7)
    np.testing.assert_allclose(
        _losses(out),
        [float(_wavy(jnp.asarray(c))) for c in calls],
        rtol=1e-6,
    )
    assert float(out[2]) == pytest.approx(np.nanmin(_losses(out)), rel=1e-6)


# =============================================================================
#                                                     4. after OpenSQP stops


def _stopped_at(calls, n):
    """The index from which every evaluation repeats the last point."""
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    return moved[-1] + 1


def test_final_point_is_re_evaluated_until_the_budget():
    calls = []
    n = 300
    out = _run(_build("opensqp", _counting(_sphere, calls)), n)
    jax.effects_barrier()
    losses = _losses(out)
    assert len(calls) == n
    stopped = _stopped_at(calls, n)
    assert stopped < n // 2
    np.testing.assert_array_equal(losses[stopped:], losses[-1])
    assert float(out[2]) < 1e-10


def test_a_run_that_stops_evaluating_is_stopped(monkeypatch):
    """At a box optimum OpenSQP keeps iterating without a new point.

    The stall guard ends the run there (ADR 0003), and the rest of the
    budget re-evaluates that point.
    """
    stalls = []
    real = opensqp_module._Stalled.__init__

    def record(self, x):
        stalls.append(x)
        real(self, x)

    monkeypatch.setattr(opensqp_module._Stalled, "__init__", record)
    calls = []
    n = 300
    out = _run(
        _build("opensqp", _counting(_sphere, calls), bounded=(-1.0, 0.2)), n
    )
    jax.effects_barrier()
    assert len(stalls) == 1
    assert _stopped_at(calls, n) < n // 2
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-6)


# =============================================================================
#                                                                 5. bounds


def test_every_evaluated_point_lies_in_the_box():
    bounded = (-1.0, 0.2)
    calls = []
    out = _run(
        _build("opensqp", _counting(_wavy, calls), bounded=bounded),
        200,
        params=2.0 + _x0(),  # start outside the box
    )
    jax.effects_barrier()
    for points in (np.stack(calls), np.asarray(out[0]), np.asarray(out[1])):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


def test_reaches_the_boundary_optimum():
    out = _run(_build("opensqp", _sphere, bounded=(-1.0, 0.2)), 300)
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-3)


# =============================================================================
#                                                    6. settings and outputs


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("maxiter", _UNREACHABLE),
        ("opt_tol", 0.0),
        ("feas_tol", 0.0),
        ("aqp_time_limit", float("inf")),
        ("turn_off_outputs", True),
    ],
)
def test_settings_reach_opensqp(monkeypatch, option, value):
    built = _spy_on_modopt(monkeypatch)
    _run(_build("opensqp"), 20)
    assert built["solver"] and built["solver"][0][option] == value


def test_writes_no_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _run(_build("opensqp", bounded=(-1.0, 0.2)), 50)
    assert list(tmp_path.iterdir()) == []


def test_leaves_the_matplotlib_backend_alone():
    """modOpt calls ``matplotlib.use("TkAgg")`` at import. With a display
    that would switch the whole process to TkAgg; without one, before
    pyplot is loaded, it breaks every later figure. A fresh interpreter,
    so modOpt is imported for the first time here."""
    code = (
        "import matplotlib\n"
        "calls = []\n"
        "matplotlib.use = lambda *args, **kwargs: calls.append(args)\n"
        "from l2co_optimizers._src.opensqp import _modopt\n"
        "_modopt()\n"
        "assert calls == [], calls\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_unknown_hyperparameters_are_rejected():
    with pytest.raises(TypeError):
        _build("opensqp", not_a_knob=1.0)


# =============================================================================
#                                              7. what is refused, and how


def test_stop_fn_raises():
    with pytest.raises(ValueError, match="stopping criterion"):
        optimizer_mapping("opensqp")(
            **quadratic_problem(jnp.zeros(DIM), _wavy),
            opt_hash=7,
            bounded=(None, None),
            stop_fn=lambda history, state: jnp.array(False),
        )


def test_has_no_single_step():
    with pytest.raises(NotImplementedError, match="owns the loop"):
        _build("opensqp").step(None, None, {})


def test_cannot_be_unpacked_into_switching_parts():
    with pytest.raises(ValueError, match="is modOpt's OpenSQP"):
        optimizer_parts(
            OptimizationStep(optimizer="opensqp"),
            **quadratic_problem(jnp.zeros(DIM), _wavy),
        )


def test_runs_realizations_one_at_a_time_as_a_gradient_method():
    update_class = _build("opensqp")
    assert isinstance(update_class, ScipyUpdateClass)
    assert update_class.sequential_realizations
    assert update_class.popsize == 1
    assert update_class.family == FAMILY_GRADIENT
    assert update_class.transfer_read_fn is None
