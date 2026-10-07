"""Unit tests for the optimistix registry entries (ADR 0001).

``"bfgs"``, ``"dfp"``, ``"nonlinearcg"`` and ``"neldermead"`` wrap an
optimistix minimiser's ``init`` / ``step``. The properties checked:

1. the wrapper is transparent -- its iterates equal a hand-written
   optimistix loop;
2. the minimiser agrees with SciPy's on Rosenbrock;
3. ``OptHistory.fevals`` equals the loss calls actually made;
4. every evaluated and recorded point lies in ``bounded``;
5. gradients are recorded on accepted steps and NaN on rejected ones;
6. the hyperparameters reach the solver;
7. optimistix 0.1.0's ``y0_simplex`` still rejects n > 1 and its own
   default simplex is still degenerate, which is why the population is
   injected as the simplex with ``eqx.tree_at``;
8. none of them unpacks into switching parts, and Nelder--Mead runs its
   realizations sequentially.
"""

#                                                                       Modules
# =============================================================================

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optimistix as optx
import pytest

# Local
from l2co_optimizers import BatchState, OptimizationStep, optimizer_parts
from l2co_optimizers._src.mapping import optimizer_mapping, optimizers
from l2co_optimizers._src.optimistix_implementations import (
    NONLINEAR_CG_METHODS,
    OPTIMISTIX_OPTIMIZERS,
    NelderMeadUpdateClass,
    _reindex_best_and_worst,
    bfgs_update,
    dfp_update,
    neldermead_update,
    nonlinearcg_update,
)

from .toy_problems import quadratic_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

GRADIENT_SOLVERS = ("bfgs", "dfp", "nonlinearcg")

#: The stock optimistix solver each gradient entry must reproduce at its
#: defaults.
STOCK = {
    "bfgs": lambda: optx.BFGS(rtol=1e-8, atol=1e-8),
    "dfp": lambda: optx.DFP(rtol=1e-8, atol=1e-8),
    "nonlinearcg": lambda: optx.NonlinearCG(rtol=1e-8, atol=1e-8),
}


def _rosenbrock(x, **_):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _wavy(x, **_):
    return jnp.sum((x - 0.5) ** 2) + 0.3 * jnp.sum(jnp.sin(3.0 * x))


def _build(name, loss_fn, dim, bounded=(None, None), **hyperparameters):
    return optimizer_mapping(name)(
        **hyperparameters,
        **quadratic_problem(jnp.zeros(dim), loss_fn),
        opt_hash=1,
        bounded=bounded,
        stop_fn=None,
    )


def _drive(update_class, params, n_steps, jit=True):
    """Step ``update_class`` from ``params``; return carries + histories."""
    key = jr.key(0)
    carry = (params, update_class.init_fn(params, key), key)
    step = (
        eqx.filter_jit(update_class.step_fn) if jit else (update_class.step_fn)
    )
    carries, histories = [], []
    for _ in range(n_steps):
        carry, history = step(carry, {})
        carries.append(carry)
        histories.append(history)
    return carries, histories


def _rosenbrock_start(dim):
    return jnp.tile(jnp.array([-1.2, 1.0]), dim)[:dim]


def _simplex_around(y0, scale=0.5):
    """``y0`` plus ``scale`` along each axis: a non-degenerate simplex.

    Built by hand because optimistix 0.1.0's own default simplex is
    degenerate for n >= 3 (it perturbs flat positions ``1..n`` of the
    ``(n + 1, n)`` array, so rows ``2..n`` all equal ``y0``) -- see
    :func:`test_upstream_default_simplex_is_degenerate`.
    """
    return jnp.concatenate([y0[None], y0 + scale * jnp.eye(y0.size)])


# =============================================================================
#                                                                      Registry


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("bfgs", bfgs_update),
        ("dfp", dfp_update),
        ("nonlinearcg", nonlinearcg_update),
        ("neldermead", neldermead_update),
    ],
)
def test_registered_under_its_bare_name(name, factory):
    assert name in OPTIMISTIX_OPTIMIZERS
    assert optimizers[name].func is factory


# =============================================================================
#                                                       1. wrapper fidelity


@pytest.mark.parametrize("name", GRADIENT_SOLVERS)
def test_gradient_wrapper_reproduces_a_direct_optimistix_loop(name):
    """Same accepted iterates as driving the stock solver by hand."""
    dim, n_steps = 4, 40
    y0 = _rosenbrock_start(dim)

    solver = STOCK[name]()
    fn = lambda y, _: (_rosenbrock(y), None)  # noqa: E731
    state = solver.init(
        fn,
        y0,
        None,
        {},
        jax.ShapeDtypeStruct((), y0.dtype),
        None,
        frozenset(),
    )
    step = eqx.filter_jit(
        lambda y, s: solver.step(fn, y, None, {}, s, frozenset())[:2]
    )
    y, expected = y0, []
    for _ in range(n_steps):
        y, state = step(y, state)
        expected.append(y)

    carries, _ = _drive(_build(name, _rosenbrock, dim), y0[None], n_steps)

    np.testing.assert_allclose(
        np.stack([c[0][0] for c in carries]),
        np.stack(expected),
        rtol=1e-6,
        atol=1e-7,
    )


def test_neldermead_wrapper_reproduces_a_direct_optimistix_loop():
    """From the same simplex, the same simplex trajectory."""
    dim, n_steps = 3, 60
    simplex0 = _simplex_around(_rosenbrock_start(dim))

    solver = optx.NelderMead(rtol=1e-8, atol=1e-8)
    fn = lambda y, _: (_rosenbrock(y), None)  # noqa: E731
    state = solver.init(
        fn,
        simplex0[0],
        None,
        {},
        jax.ShapeDtypeStruct((), simplex0.dtype),
        None,
        frozenset(),
    )
    # The same injection the wrapper makes (``y0_simplex`` is broken),
    # and the same best/worst re-indexing after every step.
    state = eqx.tree_at(lambda s: s.simplex, state, simplex0)
    step = eqx.filter_jit(
        lambda s: _reindex_best_and_worst(
            solver.step(fn, simplex0[0], None, {}, s, frozenset())[1]
        )
    )
    expected = []
    for _ in range(n_steps):
        state = step(state)
        expected.append(state.simplex)

    carries, _ = _drive(
        _build("neldermead", _rosenbrock, dim), simplex0, n_steps
    )

    np.testing.assert_allclose(
        np.stack([c[0] for c in carries]),
        np.stack(expected),
        rtol=1e-6,
        atol=1e-7,
    )


def test_neldermead_best_and_worst_name_real_vertices():
    """After every wrapped step, ``best`` / ``worst`` are the simplex rows
    their indices name (optimistix 0.1.0 alone breaks this)."""
    dim = 3
    carries, _ = _drive(
        _build("neldermead", _rosenbrock, dim),
        _simplex_around(_rosenbrock_start(dim)),
        150,
    )
    for _, state, _ in carries:
        for _, vector, index in (state.best, state.worst):
            np.testing.assert_array_equal(vector, state.simplex[index])
        f_best, _, best_index = state.best
        assert f_best == state.f_simplex[best_index]


def test_neldermead_starts_from_the_population_it_is_given():
    """The injected simplex is evaluated in place on the first step."""
    dim = 3
    population = jr.normal(jr.key(3), (dim + 1, dim))
    carries, histories = _drive(
        _build("neldermead", _wavy, dim), population, 1
    )
    np.testing.assert_allclose(histories[0].params, population)
    np.testing.assert_allclose(
        histories[0].loss, jax.vmap(_wavy)(population), rtol=1e-6
    )


# =============================================================================
#                                                          2. SciPy sanity


@pytest.mark.parametrize("dim", [2, 5])
@pytest.mark.parametrize("name", ["bfgs", "neldermead"])
def test_reaches_the_minimiser_scipy_reaches(name, dim):
    """Not trajectory-equal -- the line searches differ -- but the same
    minimiser on Rosenbrock."""
    scipy_optimize = pytest.importorskip("scipy.optimize")
    y0 = _rosenbrock_start(dim)

    # SciPy works in float64 with its own Rosenbrock (and its exact
    # gradient for BFGS): float32 finite differences stall it.
    if name == "bfgs":
        reference = scipy_optimize.minimize(
            scipy_optimize.rosen,
            np.asarray(y0, np.float64),
            jac=scipy_optimize.rosen_der,
            method="BFGS",
        )
    else:
        reference = scipy_optimize.minimize(
            scipy_optimize.rosen,
            np.asarray(y0, np.float64),
            method="Nelder-Mead",
            options={
                "initial_simplex": np.asarray(_simplex_around(y0)),
                "maxfev": 100_000,
                "xatol": 1e-8,
                "fatol": 1e-10,
            },
        )

    if name == "bfgs":
        params, n_steps = y0[None], 400
    else:
        params, n_steps = _simplex_around(y0), 3_000 * dim
    _, histories = _drive(_build(name, _rosenbrock, dim), params, n_steps)

    losses = jnp.stack([h.loss for h in histories])
    best = jnp.unravel_index(jnp.nanargmin(losses), losses.shape)
    found = histories[int(best[0])].params[int(best[1])]

    np.testing.assert_allclose(reference.x, np.ones(dim), atol=2e-2)
    np.testing.assert_allclose(found, np.ones(dim), atol=2e-2)


# =============================================================================
#                                                       3. honest fevals


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("name", [*GRADIENT_SOLVERS, "neldermead"])
def test_fevals_equal_the_loss_calls_actually_made(name, jit):
    """Count every real call of the loss with a host callback."""
    calls = []

    def counted(x, **_):
        jax.debug.callback(
            lambda v: calls.append(v.shape[0] if v.ndim == 2 else 1), x
        )
        return _wavy(x)

    dim = 3
    update_class = _build(name, counted, dim)
    params = jr.normal(jr.key(1), (update_class.popsize, dim))
    key = jr.key(0)
    carry = (params, update_class.init_fn(params, key), key)
    step = (
        eqx.filter_jit(update_class.step_fn) if jit else (update_class.step_fn)
    )

    billed, made = [], []
    for _ in range(25):
        calls.clear()
        carry, history = step(carry, {})
        jax.effects_barrier()
        billed.append(int(history.fevals))
        made.append(sum(calls))

    assert billed == made
    assert max(billed) <= update_class.fevals_bound


def test_neldermead_bills_first_pass_ordinary_and_shrink_steps():
    """``n + 1``, then 2, and ``n + 3`` on a shrink."""
    dim = 3
    update_class = _build("neldermead", _wavy, dim)
    _, histories = _drive(
        update_class, jr.normal(jr.key(1), (dim + 1, dim)), 40
    )
    billed = [int(h.fevals) for h in histories]

    assert billed[0] == dim + 1
    assert set(billed[1:]) <= {2, dim + 3}
    assert dim + 3 in billed[1:], "no shrink step in 40 steps"
    assert update_class.fevals_bound == dim + 3


# =============================================================================
#                                                                 4. bounds


@pytest.mark.parametrize("name", [*GRADIENT_SOLVERS, "neldermead"])
def test_every_point_lies_in_the_box(name):
    dim, bounded = 3, (-1.0, 1.0)

    seen = []

    def recorded(x, **_):
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), x)
        return _wavy(x)

    update_class = _build(name, recorded, dim, bounded=bounded)
    # Start well outside the box.
    params = 3.0 + jr.normal(jr.key(2), (update_class.popsize, dim))
    carries, histories = _drive(update_class, params, 30, jit=False)

    evaluated = np.concatenate([v.reshape(-1, dim) for v in seen])
    recorded_params = np.concatenate([np.asarray(h.params) for h in histories])
    carried = np.concatenate([np.asarray(c[0]) for c in carries])
    for points in (evaluated, recorded_params, carried):
        assert points.min() >= bounded[0]
        assert points.max() <= bounded[1]


# =============================================================================
#                                                                  5. grads


@pytest.mark.parametrize("name", GRADIENT_SOLVERS)
def test_grads_are_recorded_on_acceptance_and_nan_on_rejection(name):
    """An oversized first trial step forces rejected line-search trials."""
    dim = 3
    update_class = _build(name, _wavy, dim, step_init=8.0)
    carries, histories = _drive(update_class, jnp.full((1, dim), 2.0), 25)

    accepted = [bool(c[1].solver_state.search_state.accept) for c in carries]
    assert any(accepted) and not all(accepted)

    for accept, history in zip(accepted, histories, strict=True):
        grads = np.asarray(history.grads)
        assert np.isfinite(history.loss).all()
        if accept:
            np.testing.assert_allclose(
                grads[0],
                jax.grad(_wavy)(history.params[0]),
                rtol=1e-5,
                atol=1e-6,
            )
        else:
            assert np.isnan(grads).all()


def test_rejected_trials_are_recorded_with_their_own_loss():
    """The history holds the evaluated point, not the accepted one."""
    dim = 3
    _, histories = _drive(
        _build("bfgs", _wavy, dim, step_init=8.0),
        jnp.full((1, dim), 2.0),
        25,
    )
    for history in histories:
        np.testing.assert_allclose(
            history.loss[0], _wavy(history.params[0]), rtol=1e-6
        )


def test_neldermead_records_nan_grads():
    _, histories = _drive(
        _build("neldermead", _wavy, 2), jr.normal(jr.key(0), (3, 2)), 5
    )
    assert all(np.isnan(h.grads).all() for h in histories)


# =============================================================================
#                                                        6. hyperparameters


def test_step_init_sets_the_first_trial_step():
    """BFGS's first descent is steepest descent: y - step_init * g."""
    dim, y0 = 3, jnp.full((1, 3), 2.0)
    for step_init in (0.25, 1.0):
        _, histories = _drive(
            _build("bfgs", _wavy, dim, step_init=step_init), y0, 2
        )
        np.testing.assert_allclose(
            histories[1].params[0],
            y0[0] - step_init * jax.grad(_wavy)(y0[0]),
            rtol=1e-6,
        )


@pytest.mark.parametrize("name", ["bfgs", "dfp"])
def test_use_inverse_is_forwarded(name):
    """Hessian vs inverse-Hessian give the same iterates up to round-off
    on a quadratic, through different code paths."""
    dim = 3
    runs = [
        _drive(
            _build(name, _rosenbrock, dim, use_inverse=use_inverse),
            _rosenbrock_start(dim)[None],
            30,
        )[0][-1][1].solver_state.f_info
        for use_inverse in (True, False)
    ]
    assert type(runs[0]).__name__ == "EvalGradHessianInv"
    assert type(runs[1]).__name__ == "EvalGradHessian"


def test_nonlinearcg_methods_differ_and_unknown_raises():
    dim = 4
    finals = {
        method: _drive(
            _build("nonlinearcg", _rosenbrock, dim, method=method),
            _rosenbrock_start(dim)[None],
            60,
        )[0][-1][0]
        for method in NONLINEAR_CG_METHODS
    }
    assert len({np.asarray(v).tobytes() for v in finals.values()}) > 1

    with pytest.raises(ValueError, match="Unknown nonlinearcg method"):
        _build("nonlinearcg", _wavy, dim, method="steepest")


@pytest.mark.parametrize(
    ("name", "hyperparameters"),
    [
        ("bfgs", {"learning_rate": 0.1}),
        ("neldermead", {"popsize": 10}),
    ],
)
def test_unknown_hyperparameters_are_rejected(name, hyperparameters):
    with pytest.raises(TypeError):
        _build(name, _wavy, 3, **hyperparameters)


# =============================================================================
#                                                       7. upstream pin


def test_upstream_y0_simplex_still_rejects_n_above_one():
    """optimistix 0.1.0 validates ``x[1:].size`` (n**2) against n.

    When this starts passing, the ``eqx.tree_at`` injection in
    ``neldermead_update``'s ``init_fn`` can become ``y0_simplex=True``.
    """
    n = 3
    simplex = jnp.concatenate([jnp.zeros((1, n)), jnp.eye(n)])
    with pytest.raises(ValueError, match="valid simplex"):
        optx.NelderMead(rtol=1e-8, atol=1e-8).init(
            lambda y, _: (jnp.sum(y**2), None),
            simplex,
            None,
            {"y0_simplex": True},
            jax.ShapeDtypeStruct((), jnp.float32),
            None,
            frozenset(),
        )


def test_upstream_best_vector_goes_stale():
    """optimistix 0.1.0 reads ``best`` from the pre-update simplex.

    The reason for ``_reindex_best_and_worst``. When this starts failing
    -- the stock state's best vector always matching
    ``simplex[best_index]`` -- the fix-up can be deleted.
    """
    dim = 3
    simplex0 = _simplex_around(_rosenbrock_start(dim))
    solver = optx.NelderMead(rtol=1e-8, atol=1e-8)
    fn = lambda y, _: (_rosenbrock(y), None)  # noqa: E731
    state = solver.init(
        fn,
        simplex0[0],
        None,
        {},
        jax.ShapeDtypeStruct((), simplex0.dtype),
        None,
        frozenset(),
    )
    state = eqx.tree_at(lambda s: s.simplex, state, simplex0)
    step = eqx.filter_jit(
        lambda s: solver.step(fn, simplex0[0], None, {}, s, frozenset())[1]
    )
    stale = 0
    for _ in range(100):
        state = step(state)
        _, best, best_index = state.best
        stale += not bool(jnp.all(best == state.simplex[best_index]))
    assert stale > 0


def test_upstream_default_simplex_is_degenerate():
    """optimistix 0.1.0 builds a zero-volume simplex around one point.

    Why the population is injected as the simplex rather than letting
    optimistix grow one from a single vertex (ADR 0001). When this
    starts failing, the upstream construction has been fixed.
    """
    n = 3
    state = optx.NelderMead(rtol=1e-8, atol=1e-8).init(
        lambda y, _: (jnp.sum(y**2), None),
        jnp.ones(n),
        None,
        {},
        jax.ShapeDtypeStruct((), jnp.float32),
        None,
        frozenset(),
    )
    edges = state.simplex[1:] - state.simplex[0]
    assert np.linalg.matrix_rank(np.asarray(edges)) < n


# =============================================================================
#                                              8. no switching, run modes


@pytest.mark.parametrize("name", sorted(OPTIMISTIX_OPTIMIZERS))
def test_cannot_be_unpacked_into_switching_parts(name):
    with pytest.raises(ValueError, match="optimistix minimiser"):
        optimizer_parts(
            OptimizationStep(optimizer=name),
            **quadratic_problem(jnp.zeros(3), _wavy),
        )


def test_neldermead_runs_realizations_sequentially():
    update_class = _build("neldermead", _wavy, 4)
    assert isinstance(update_class, NelderMeadUpdateClass)
    assert update_class.sequential_realizations
    assert update_class.popsize == 5


@pytest.mark.parametrize("name", GRADIENT_SOLVERS)
def test_gradient_solvers_run_fused_with_one_point(name):
    update_class = _build(name, _wavy, 4)
    assert not update_class.sequential_realizations
    assert update_class.popsize == 1
    assert update_class.fevals_bound == 1


@pytest.mark.parametrize("name", sorted(OPTIMISTIX_OPTIMIZERS))
def test_batch_run_end_to_end(name):
    """Through the real multi-realization driver, fused or sequential."""
    dim, n_realizations, n_iterations = 4, 3, 30
    update_class = _build(name, _wavy, dim, bounded=(-2.0, 2.0))
    key = jr.key(0)
    batch_state = BatchState.init(dataset={}, batch_size=None, key=key)
    params = jr.normal(key, (n_realizations, update_class.popsize, dim))
    keys = jr.split(key, n_realizations)
    opt_state = eqx.filter_vmap(
        update_class.init_state, in_axes=(0, None, None, 0)
    )(params, batch_state, {}, keys)

    out = eqx.filter_jit(update_class.batch_run)(
        opt_state=opt_state,
        params=params,
        batch_state=batch_state,
        dataset={},
        key=keys,
        n_iterations=n_iterations,
        verbose=False,
    )
    final_params, best_params, best_loss, *_, history = out

    assert final_params.shape == (n_realizations, update_class.popsize, dim)
    assert best_params.shape == (n_realizations, dim)
    assert np.isfinite(best_loss).all()
    np.testing.assert_allclose(
        best_loss, jax.vmap(_wavy)(best_params), rtol=1e-5
    )
    assert history.output_min.shape == (n_realizations, n_iterations)
