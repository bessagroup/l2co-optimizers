"""Unit tests for the IPOPT registry entry (ADR 0003).

``"ipopt"`` hands a whole run to IPOPT, through casadi, inside one host
callback, reusing the scipy entries' driver (ADR 0002). The properties
checked are those of ``test_scipy.py``, plus two of IPOPT's own:

1. the wrapper is transparent -- its evaluations are the ones a direct
   casadi IPOPT run makes, until IPOPT stops; and IPOPT's separate
   value and gradient requests at one point are one evaluation;
2. it improves on its starting point;
3. one iteration is one real evaluation, billed one, for exactly
   ``n_iterations``, even though IPOPT keeps asking after the budget;
4. after IPOPT stops, its final point is re-evaluated to the budget;
5. every evaluated point lies in ``bounded``;
6. the hyperparameters reach IPOPT;
7. ``stop_fn`` raises, ``step`` raises, it does not unpack into switching
   parts, and realizations run one at a time.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    FAMILY_GRADIENT,
    OptimizationStep,
    optimizer_parts,
)
from l2co_optimizers._src.ipopt import (
    _IPOPT_TOL,
    _UNREACHABLE,
    IPOPT_OPTIMIZERS,
    ipopt_update,
)
from l2co_optimizers._src.mapping import (
    optimizer_mapping,
    optimizers,
)
from l2co_optimizers._src.scipy_implementations import (
    ScipyUpdateClass,
)

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

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _sphere(x, **_):
    return jnp.sum((x - 0.5) ** 2)


def test_registered_under_its_bare_name():
    assert "ipopt" in IPOPT_OPTIMIZERS
    assert optimizers["ipopt"].func is ipopt_update


# =============================================================================
#                                                1. transparent wrapper


def _direct_ipopt(loss_fn, x0):
    """The losses a direct casadi IPOPT run evaluates, one per new point.

    An independent re-statement of the wrapper: IPOPT asks for the value
    and the gradient separately, and only the first request at a point
    is recorded.
    """
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    seen, points, alive = [], {}, []

    def at(x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.tobytes() not in points:
            v, g = value_and_grad(jnp.asarray(x))
            points[x.tobytes()] = (float(v), np.asarray(g, dtype=np.float64))
            seen.append(float(v))
        return points[x.tobytes()]

    class Gradient(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("g", {})

        def get_n_in(self):
            return 2

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1) if i == 0 else ca.Sparsity.scalar()

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(1, n)

        def eval(self, arg):
            return [ca.DM(at(arg[0].full())[1]).T]

    class Loss(ca.Callback):
        def __init__(self):
            ca.Callback.__init__(self)
            self.construct("f", {})

        def get_n_in(self):
            return 1

        def get_n_out(self):
            return 1

        def get_sparsity_in(self, i):
            return ca.Sparsity.dense(n, 1)

        def get_sparsity_out(self, i):
            return ca.Sparsity.scalar()

        def eval(self, arg):
            return [at(arg[0].full())[0]]

        def has_jacobian(self):
            return True

        def get_jacobian(self, *args):
            alive.append(Gradient())  # casadi frees an unreferenced one
            return alive[-1]

    loss = Loss()
    x = ca.MX.sym("x", n)
    solver = ca.nlpsol(
        "direct",
        "ipopt",
        {"x": x, "f": loss(x)},
        {
            "print_time": False,
            "error_on_fail": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.hessian_approximation": "limited-memory",
            "ipopt.tol": _IPOPT_TOL,
            "ipopt.acceptable_iter": 0,
            "ipopt.max_iter": 300,
        },
    )
    solver(x0=x0)
    return np.asarray(seen)


def test_evaluations_match_a_direct_ipopt_run():
    x0 = _x0()
    direct = _direct_ipopt(_wavy, x0[0])
    n = 400
    ours = _losses(_run(_build("ipopt"), n, params=x0))
    k = min(len(direct), n)
    assert k > 10, f"IPOPT stopped after {k} evaluations"
    np.testing.assert_allclose(ours[:k], direct[:k], rtol=1e-6)


# From seed 1 IPOPT stops on its own at evaluation 173; from seed 2 it is
# still moving when the 400-evaluation budget runs out. Where it stops
# turns on its last machine-precision step, so which case a start lands
# in can differ between platforms (seed 1 runs out on macOS CI). Both
# cases are covered, and the claim holds in either.
@pytest.mark.parametrize("seed", [1, 2], ids=["stops", "runs-out"])
def test_repeated_requests_at_one_point_are_one_evaluation(monkeypatch, seed):
    """IPOPT asks for the value and the gradient separately, and often
    asks again at the point it just evaluated; each point is billed once,
    as scipy's own cache of the latest point does for the scipy entries.
    """
    solvers = []
    real = ca.nlpsol

    def spy(*args, **kwargs):
        solvers.append(real(*args, **kwargs))
        return solvers[-1]

    monkeypatch.setattr(ca, "nlpsol", spy)
    calls = []
    n = 400
    _run(_build("ipopt", _counting(_wavy, calls)), n, params=_x0(seed=seed))
    jax.effects_barrier()
    stats = solvers[0].stats()
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    # Billed evaluations up to IPOPT's last new point; if it stopped before
    # the budget, the rest of the budget repeats its final point.
    billed = moved[-1] + 1
    assert stats["n_call_nlp_grad_f"] > 0
    assert billed < stats["n_call_nlp_f"]


# =============================================================================
#                                                 2. and 3. progress, billing


def test_improves_on_its_starting_point():
    out = _run(_build("ipopt"), 300, params=3.0 + _x0())
    assert float(out[2]) < 0.2 * _losses(out)[0]


def test_each_iteration_is_one_real_evaluation():
    calls = []
    n = 120
    out = _run(_build("ipopt", _counting(_wavy, calls)), n)
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
#                                                       4. after IPOPT stops


def test_final_point_is_re_evaluated_until_the_budget():
    calls = []
    n = 300
    out = _run(_build("ipopt", _counting(_sphere, calls)), n)
    jax.effects_barrier()
    losses = _losses(out)
    assert len(calls) == n
    moved = [i for i in range(n) if not np.array_equal(calls[i], calls[-1])]
    stopped = moved[-1] + 1
    assert stopped < n // 2
    np.testing.assert_array_equal(losses[stopped:], losses[-1])
    assert float(out[2]) < 1e-10


# =============================================================================
#                                                                 5. bounds


def test_every_evaluated_point_lies_in_the_box():
    bounded = (-1.0, 0.2)
    calls = []
    out = _run(
        _build("ipopt", _counting(_wavy, calls), bounded=bounded),
        200,
        params=2.0 + _x0(),  # start outside the box
    )
    jax.effects_barrier()
    for points in (np.stack(calls), np.asarray(out[0]), np.asarray(out[1])):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


def test_reaches_the_boundary_optimum():
    out = _run(_build("ipopt", _sphere, bounded=(-1.0, 0.2)), 300)
    np.testing.assert_allclose(np.asarray(out[1]), 0.2, atol=1e-3)


# =============================================================================
#                                                        6. hyperparameters


@pytest.mark.parametrize(
    ("hp", "option", "value"),
    [
        ({}, "ipopt.limited_memory_max_history", 6),
        (
            {"limited_memory_max_history": 2},
            "ipopt.limited_memory_max_history",
            2,
        ),
        ({"mu_init": 0.5}, "ipopt.mu_init", 0.5),
        ({}, "ipopt.max_iter", _UNREACHABLE),
        ({}, "ipopt.bound_relax_factor", 0.0),
    ],
)
def test_hyperparameters_reach_ipopt(monkeypatch, hp, option, value):
    seen = []
    real = ca.nlpsol

    def spy(*args, **kwargs):
        seen.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(ca, "nlpsol", spy)
    _run(_build("ipopt", **hp), 20)
    assert seen and seen[0][option] == value


def test_history_length_changes_the_run():
    def ellipsoid(x, **_):
        return jnp.sum(jnp.array([1.0, 1e2, 1e4]) * (x - 0.5) ** 2)

    params = 3.0 + _x0()
    default = _losses(_run(_build("ipopt", ellipsoid), 60, params=params))
    short = _losses(
        _run(
            _build("ipopt", ellipsoid, limited_memory_max_history=1),
            60,
            params=params,
        )
    )
    assert not np.array_equal(default, short)


def test_unknown_hyperparameters_are_rejected():
    with pytest.raises(TypeError):
        _build("ipopt", not_a_knob=1.0)


# =============================================================================
#                                              7. what is refused, and how


def test_stop_fn_raises():
    with pytest.raises(ValueError, match="stopping criterion"):
        optimizer_mapping("ipopt")(
            **quadratic_problem(jnp.zeros(DIM), _wavy),
            opt_hash=7,
            bounded=(None, None),
            stop_fn=lambda history, state: jnp.array(False),
        )


def test_has_no_single_step():
    with pytest.raises(NotImplementedError, match="owns the loop"):
        _build("ipopt").step(None, None, {})


def test_cannot_be_unpacked_into_switching_parts():
    with pytest.raises(ValueError, match="is IPOPT"):
        optimizer_parts(
            OptimizationStep(optimizer="ipopt"),
            **quadratic_problem(jnp.zeros(DIM), _wavy),
        )


def test_runs_realizations_one_at_a_time_as_a_gradient_method():
    update_class = _build("ipopt")
    assert isinstance(update_class, ScipyUpdateClass)
    assert update_class.sequential_realizations
    assert update_class.popsize == 1
    assert update_class.family == FAMILY_GRADIENT
    assert update_class.transfer_read_fn is None
