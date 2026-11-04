"""``ipopt`` and ``trustconstr`` with ``hessian="exact"`` (ADR 0007).

The properties checked:

1. the wrapper is transparent: its history is a direct casadi IPOPT or
   scipy trust-constr run's sequence of evaluations, each Hessian one
   entry recording the loss where it was taken;
2. a Hessian is taken on the sample its point was evaluated with;
3. exact Newton beats the quasi-Newton default on an ill-conditioned
   quadratic, even with its Hessians billed;
4. the box holds, a budget that runs out inside a Hessian ends the run
   cleanly, and a loss JAX cannot differentiate twice is refused loudly;
5. the hyperparameters are checked and the defaults are unchanged.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import casadi as ca
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.optimize as so

# Local
from l2co_optimizers import BatchState, OptimizationStep
from l2co_optimizers._src.mapping import optimizer_mapping
from l2co_optimizers._src.scipy_implementations import (
    _TRUSTCONSTR_GTOL,
    _TRUSTCONSTR_XTOL,
    _UNREACHABLE,
    _Driver,
)

from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

NAMES = ("ipopt", "trustconstr")
DIM = 3


def _wavy(x, **_):
    return jnp.sum((x - 0.5) ** 2) + 0.3 * jnp.sum(jnp.sin(3.0 * x))


def _ellipsoid(x, **_):
    w = 1e6 ** (jnp.arange(x.shape[0]) / (x.shape[0] - 1))
    return jnp.sum(w * (x - 0.5) ** 2)


def _build(name, loss_fn=_wavy, dim=DIM, bounded=None, pass_rng=False, **hp):
    return optimizer_mapping(name)(
        **hp,
        **quadratic_problem(jnp.zeros(dim), loss_fn, pass_rng),
        opt_hash=1,
        bounded=bounded,
        stop_fn=None,
    )


def _x0(dim=DIM, seed=1):
    return jr.normal(jr.key(seed), (1, dim))


def _run(update_class, n_iterations, params=None, key=None):
    key = jr.key(0) if key is None else key
    params = _x0(update_class_dim(update_class)) if params is None else params
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    opt_state = update_class.init_state(params, batch_state, {}, key)
    return update_class.run(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
        key=key,
        n_iterations=n_iterations,
        verbose=False,
    )


def update_class_dim(_):
    return DIM


def _losses(out):
    return np.asarray(out[-1].output_min)


# =============================================================================
#                                              1. the same evaluations


def _direct_trustconstr(loss_fn, x0):
    """A direct trust-constr run with the exact Hessian, as the wrapper
    records it: each value-and-gradient call and each Hessian call is one
    entry, the Hessian's recording the loss where it was taken."""
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    hessian = jax.jit(jax.hessian(loss_fn))
    seen = []

    def fg(x):
        v, g = value_and_grad(jnp.asarray(x))
        seen.append(float(v))
        return float(v), np.asarray(g, dtype=np.float64)

    def h(x):
        seen.append(float(loss_fn(jnp.asarray(x))))
        return np.asarray(hessian(jnp.asarray(x)), dtype=np.float64)

    so.minimize(
        fg,
        np.asarray(x0, dtype=np.float64),
        jac=True,
        hess=h,
        method="trust-constr",
        options=dict(
            gtol=_TRUSTCONSTR_GTOL,
            xtol=_TRUSTCONSTR_XTOL,
            maxiter=_UNREACHABLE,
        ),
    )
    return np.asarray(seen)


def _direct_ipopt(loss_fn, x0):
    """A direct casadi IPOPT run with the exact Hessian, recorded the
    same way; IPOPT's repeated value and gradient requests at one point
    are one entry."""
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    hessian = jax.jit(jax.hessian(loss_fn))
    x0 = np.asarray(x0, dtype=np.float64)
    n = x0.size
    seen, points, alive = [], {}, []

    def at(x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.tobytes() not in points:
            v, g = value_and_grad(jnp.asarray(x))
            points.clear()
            points[x.tobytes()] = (float(v), np.asarray(g, dtype=np.float64))
            seen.append(float(v))
        return points[x.tobytes()]

    class Hessian(ca.Callback):
        def __init__(self, inames, onames):
            ca.Callback.__init__(self)
            self.names = list(inames), list(onames)
            self.construct("h", {})

        def get_n_in(self):
            return 3

        def get_n_out(self):
            return 2

        def get_name_in(self, i):
            return self.names[0][i]

        def get_name_out(self, i):
            return self.names[1][i]

        def get_sparsity_in(self, i):
            return [
                ca.Sparsity.dense(n, 1),
                ca.Sparsity.scalar(),
                ca.Sparsity.dense(1, n),
            ][i]

        def get_sparsity_out(self, i):
            return ca.Sparsity.dense(n, n) if i == 0 else ca.Sparsity(n, 1)

        def eval(self, arg):
            x = jnp.asarray(arg[0].full().ravel())
            seen.append(float(loss_fn(x)))
            return [ca.DM(np.asarray(hessian(x), np.float64)), ca.DM(n, 1)]

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

        def has_jacobian(self):
            return True

        def get_jacobian(self, name, inames, onames, opts):
            h = Hessian(inames, onames)
            alive.append(h)
            return h

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

        def get_jacobian(self, name, inames, onames, opts):
            g = Gradient()
            alive.append(g)
            return g

    loss = Loss()
    alive.append(loss)
    x = ca.MX.sym("x", n)
    solver = ca.nlpsol(
        "ipopt",
        "ipopt",
        {"x": x, "f": loss(x)},
        {
            "error_on_fail": False,
            "show_eval_warnings": False,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.hessian_approximation": "exact",
            "ipopt.tol": 1e-300,
            "ipopt.acceptable_iter": 0,
            "ipopt.max_iter": 3000,
            "ipopt.bound_relax_factor": 0.0,
            "ipopt.mu_init": 0.1,
        },
    )
    solver(x0=x0, lbx=np.full(n, -np.inf), ubx=np.full(n, np.inf))
    return np.asarray(seen)


@pytest.mark.parametrize(
    ("name", "direct"),
    [("trustconstr", _direct_trustconstr), ("ipopt", _direct_ipopt)],
)
def test_evaluations_match_a_direct_run(name, direct):
    x0 = _x0()
    reference = direct(_wavy, x0[0])
    n = 300
    ours = _losses(_run(_build(name, hessian="exact"), n, params=x0))
    k = min(len(reference), n)
    assert k > 10, f"{name} stopped after {k} evaluations"
    np.testing.assert_allclose(ours[:k], reference[:k], rtol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_each_hessian_is_one_entry(monkeypatch, name):
    """Hessians and evaluations together fill the budget exactly."""
    counts = {"hessian": 0, "value_and_grad": 0}
    for method in counts:
        real = getattr(_Driver, method)

        def spy(self, x, _real=real, _method=method):
            counts[_method] += 1
            return _real(self, x)

        monkeypatch.setattr(_Driver, method, spy)
    n = 60
    out = _run(_build(name, hessian="exact"), n)
    assert counts["hessian"] > 0
    assert np.isfinite(_losses(out)).all()
    assert out[-1].output_min.shape == (n,)


# =============================================================================
#                                                      2. the same sample


@pytest.mark.parametrize("name", NAMES)
def test_a_hessian_is_taken_on_its_points_sample(monkeypatch, name):
    """``f = s |x - 0.5|^2`` and ``H = 2 s I``, with ``s`` drawn from the
    key: on the same sample, ``H[0, 0] |x - 0.5|^2 / 2 == f(x)``."""

    def noisy(x, *, key, **_):
        scale = 1.0 + 0.5 * jnp.abs(jr.normal(key))
        return scale * jnp.sum((x - 0.5) ** 2)

    evaluated, checked = {}, []
    real_vg, real_h = _Driver.value_and_grad, _Driver.hessian

    def vg(self, x):
        f, g = real_vg(self, x)
        evaluated[np.asarray(x, np.float64).tobytes()] = f
        return f, g

    def h(self, x):
        hessian = real_h(self, x)
        x = np.asarray(x, np.float64)
        f = evaluated.get(x.tobytes())
        if f is not None:
            r2 = float(np.sum((x - 0.5) ** 2))
            checked.append((hessian[0, 0] * r2 / 2, f))
            # The entry it adds records that same loss.
            assert self.losses[self.count - 1] == pytest.approx(f, rel=1e-6)
        return hessian

    monkeypatch.setattr(_Driver, "value_and_grad", vg)
    monkeypatch.setattr(_Driver, "hessian", h)
    _run(_build(name, loss_fn=noisy, pass_rng=True, hessian="exact"), 40)
    assert checked
    for from_hessian, loss in checked:
        assert from_hessian == pytest.approx(loss, rel=1e-5)


# =============================================================================
#                                                 3. what an exact one buys


@pytest.mark.parametrize(
    ("name", "at_most"),
    # IPOPT takes one Newton step: value and gradient, Hessian, then the
    # minimum. trust-constr's radius starts at 1 and has to grow first.
    [("ipopt", 3), ("trustconstr", 50)],
)
def test_beats_the_quasi_newton_default_on_an_ill_conditioned_quadratic(
    name, at_most
):
    with jax.enable_x64(True):
        params = jnp.zeros((1, 10))
        exact = _losses(
            _run(_build(name, _ellipsoid, 10, hessian="exact"), 150, params)
        )
        default = _losses(_run(_build(name, _ellipsoid, 10), 150, params))

    def evals_to(losses, target=1e-10):
        hit = np.nonzero(np.minimum.accumulate(losses) <= target)[0]
        return int(hit[0]) + 1 if len(hit) else np.inf

    assert evals_to(exact) <= at_most
    assert evals_to(exact) < evals_to(default)


# =============================================================================
#                                          4. box, budget, refused losses


@pytest.mark.parametrize("name", NAMES)
def test_every_requested_point_lies_in_the_box(monkeypatch, name):
    bounded, requested = (-1.0, 0.2), []
    real = _Driver._check

    def check(self, x):
        requested.append(np.asarray(x))
        return real(self, x)

    monkeypatch.setattr(_Driver, "_check", check)
    _run(_build(name, bounded=bounded, hessian="exact"), 40)
    points = np.stack(requested)
    assert points.min() >= bounded[0] and points.max() <= bounded[1]


@pytest.mark.parametrize("n_iterations", [1, 2, 3, 7])
@pytest.mark.parametrize("name", NAMES)
def test_a_budget_that_runs_out_mid_run_ends_cleanly(name, n_iterations):
    out = _run(_build(name, hessian="exact"), n_iterations)
    assert out[-1].output_min.shape == (n_iterations,)
    assert np.isfinite(_losses(out)).all()


def _host_loss(x, **_):
    """A loss computed outside JAX, with a first-order rule only, as
    l2co-tasks' host objectives had before ADR 0007."""

    @jax.custom_jvp
    def f(z):
        return jax.pure_callback(
            lambda v: np.asarray(np.sum(v**2), v.dtype),
            jax.ShapeDtypeStruct((), z.dtype),
            z,
        )

    @f.defjvp
    def f_jvp(primals, tangents):
        (z,), (t,) = primals, tangents
        g = jax.pure_callback(
            lambda v: np.asarray(2 * v, v.dtype),
            jax.ShapeDtypeStruct(z.shape, z.dtype),
            z,
        )
        return f(z), jnp.dot(g, t)

    return f(x)


@pytest.mark.parametrize("name", NAMES)
def test_a_loss_without_second_derivatives_is_refused(name):
    with pytest.raises(Exception, match="differentiated twice"):
        _run(_build(name, loss_fn=_host_loss, hessian="exact"), 10)


@pytest.mark.parametrize("name", NAMES)
def test_the_default_still_runs_on_such_a_loss(name):
    out = _run(_build(name, loss_fn=_host_loss), 30)
    assert np.isfinite(_losses(out)).all()


# =============================================================================
#                                                       5. hyperparameters


@pytest.mark.parametrize(
    ("name", "hp"),
    [
        ("ipopt", {"hessian": "bfgs"}),
        ("trustconstr", {"hessian": "limited-memory"}),
        ("ipopt", {"hessian": "exact", "limited_memory_max_history": 4}),
    ],
)
def test_bad_settings_are_refused(name, hp):
    with pytest.raises(ValueError, match="hessian"):
        _build(name, **hp)


def test_ipopt_default_history_still_reaches_ipopt(monkeypatch):
    import l2co_optimizers._src.ipopt as ipopt_module

    seen = []
    real = ipopt_module._ipopt_minimize

    def spy(*args, **kwargs):
        seen.append((kwargs["options"], kwargs["exact_hessian"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(ipopt_module, "_ipopt_minimize", spy)
    _run(_build("ipopt"), 10)
    _run(_build("ipopt", limited_memory_max_history=3), 10)
    _run(_build("ipopt", hessian="exact"), 10)
    assert seen == [
        ({"mu_init": 0.1, "limited_memory_max_history": 6}, False),
        ({"mu_init": 0.1, "limited_memory_max_history": 3}, False),
        ({"mu_init": 0.1}, True),
    ]


@pytest.mark.parametrize("name", NAMES)
def test_schedule_names(name):
    assert OptimizationStep(optimizer=name).name == name
    step = OptimizationStep(
        optimizer=name, hyperparameters={"hessian": "exact"}
    )
    assert step.name == f"{name}_hessian=exact"
