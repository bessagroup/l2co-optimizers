"""Unit tests for the optimizer state-transfer channel.

The channel carries one scalar -- a search *scale* -- from the outgoing
optimizer to the incoming one at a switch. These tests pin the four
things that can quietly go wrong with that:

* the **round trip**: a writer must be the inverse of the reader, or a
  transferred scale silently lands at the wrong magnitude. Two readers
  fold in a shape factor (an ES's covariance, an optax preconditioner),
  which is exactly where an inverse is easy to get wrong;
* the **roles**: not every optimizer both provides and receives. L-BFGS
  and DE can be read but not written; RMSProp can be written but not
  read. Pinning the asymmetry stops a future refactor from "fixing" a
  deliberate no-op;
* the **gate**: a bundle carrying no scale must leave state untouched,
  and a degenerate scale (zero, NaN) must be reported as absence rather
  than written -- writing zero into an ES freezes it;
* **shape preservation**: a per-coordinate quantity is rescaled, never
  flattened. Writing a scalar into ``rprop``'s ``step_sizes`` would
  destroy the ratchet that makes it work.

Plus the two structural guards: every registered optimizer's reader must
be usable in one ``lax.switch`` (they all have to agree on output
layout), and every optimizer whose ``TransferSpec.own_ask`` is set must
expose an ``ask_fn``, since a missing one silently degrades to the old
population-substitution path rather than raising.

Two more things are pinned here because they are decisions rather than
mechanics: *which* optimizers draw their own post-switch population
(a per-optimizer property, not a family one), and that the repaired
``state.fitness`` baseline claims a measurement -- valid on a
deterministic loss, false on a stochastic one, where claiming it
collapses the mutation scale. See ``docs/adr/0014``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from evosax.algorithms.distribution_based import (
    CMA_ES,
    CR_FM_NES,
    SNES,
    Sep_CMA_ES,
)
from evosax.algorithms.population_based import (
    MR15_GA,
    PSO,
    DifferentialEvolution,
    SimpleGA,
)

# Local
from l2co_optimizers import (
    CONF_ABSENT,
    CONF_EXACT,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    TransferBundle,
    build_transfer_fns,
)
from l2co_optimizers import optimizers as REGISTRY
from l2co_optimizers._src.state_transfer import (
    TRANSFER_OVERRIDES,
    transfer_spec_for,
)

from .toy_problems import sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 10
POPSIZE = 10
TARGET = 0.25

#: The nine-member transfer-validation menu, with the hyperparameters
#: the configs use. Covers every provider class -- line-search,
#: sign-based, diag+rank-1 ES, diagonal ES, gradient second-moment,
#: population/archive -- and all three mutation GAs that draw their own
#: post-switch population (``docs/adr/0014``).
MENU: dict[str, dict] = {
    "lbfgs": {},
    "rprop": {"learning_rate": 1e-3},
    "crfmnes": {},
    "sepcmaes": {},
    "adan": {"learning_rate": 1e-2},
    "pso": {},
    "mr15ga": {},
    "samrga": {},
    "gesmrga": {},
}


# ---------------------------------------------------------------- helpers


def _bundle(sigma: float = TARGET, conf: float = CONF_EXACT):
    """A bundle handing over a known point at a known scale."""
    return TransferBundle(
        x=jnp.full((DIM,), 0.5),
        f=jnp.asarray(-1.0),
        sigma=jnp.asarray(sigma),
        conf_sigma=jnp.asarray(conf),
    )


def _evosax_dist(cls, **hp):
    """Build a distribution-based ES and its reader/writer."""
    algo = cls(population_size=POPSIZE, solution=jnp.zeros((DIM,)))
    state = algo.init(jr.key(0), jnp.zeros((DIM,)), algo.default_params)
    read, write = build_transfer_fns(
        cls.__name__.lower().replace("_", ""),
        FAMILY_DISTRIBUTION,
        hp,
        ravel_fn=algo._ravel_solution,
    )
    return algo, state, read, write


def _evosax_pop(cls, name, generations: int = 0):
    """Build a population-based ES, optionally after real generations.

    PSO's ``velocity`` and DE's population spread only become meaningful
    once a generation has actually run, so scale is genuinely absent at
    init -- which some of these tests rely on and others must step past.
    """
    key = jr.key(0)
    algo = cls(population_size=POPSIZE, solution=jnp.zeros((DIM,)))
    state = algo.init(
        key,
        jr.normal(key, (POPSIZE, DIM)),
        jnp.arange(POPSIZE, dtype=float),
        algo.default_params,
    )
    for _ in range(generations):
        population, state = algo.ask(key, state, algo.default_params)
        state, _ = algo.tell(
            key,
            population,
            jnp.arange(POPSIZE, dtype=float),
            state,
            algo.default_params,
        )
    read, write = build_transfer_fns(name, FAMILY_POPULATION, {})
    return algo, state, read, write


def _optax_stepped(transform, steps: int = 3):
    """An optax state after real updates, so its moments are non-zero."""
    params = jnp.ones((1, DIM))
    state = transform.init(params)
    for i in range(steps):
        grads = jnp.full((1, DIM), 0.1 * (i + 1))
        updates, state = transform.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return state


# ------------------------------------------------------------ round trip


class TestRoundTrip:
    """A writer must be the exact inverse of its reader.

    Otherwise a scale handed over as ``sigma`` arrives as something
    else. The ES cases are the ones that matter: their readers multiply
    ``std`` by a shape factor drawn from ``C`` / ``D`` / ``v``, and the
    writer has to divide the *target's own* factor back out -- the
    target keeps its shape across a switch, so the factor is not the
    one the reader saw.
    """

    @pytest.mark.parametrize(
        "cls", [CR_FM_NES, Sep_CMA_ES, SNES, CMA_ES], ids=lambda c: c.__name__
    )
    def test_distribution_round_trip(self, cls):
        _, state, read, write = _evosax_dist(cls)
        sigma, conf = read(write(_bundle(), state))
        assert float(conf) > CONF_ABSENT
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    def test_mr15ga_round_trip(self):
        _, state, read, write = _evosax_pop(MR15_GA, "mr15ga")
        sigma, _ = read(write(_bundle(), state))
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    def test_pso_round_trip(self):
        # One generation, so ``velocity`` is no longer zero.
        _, state, read, write = _evosax_pop(PSO, "pso", generations=1)
        sigma, _ = read(write(_bundle(), state))
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    @pytest.mark.parametrize(
        ("name", "transform", "hp"),
        [
            ("adam", optax.adam(1e-2), {"learning_rate": 1e-2}),
            ("adan", optax.adan(1e-2), {"learning_rate": 1e-2}),
            ("sgd", optax.sgd(1e-2, momentum=0.9), {"learning_rate": 1e-2}),
            ("rprop", optax.rprop(1e-3), {"learning_rate": 1e-3}),
        ],
    )
    def test_gradient_round_trip(self, name, transform, hp):
        read, write = build_transfer_fns(name, FAMILY_GRADIENT, hp)
        state = _optax_stepped(transform)
        sigma, conf = read(write(_bundle(), state))
        assert float(conf) > CONF_ABSENT
        # Looser than the ES cases: the ``eps`` in ``mu / (sqrt(nu) +
        # eps)`` does not scale with ``nu``, so the inverse is exact
        # only up to that term. It is why these readers report
        # ``CONF_ESTIMATED``.
        assert jnp.isclose(sigma, TARGET, rtol=1e-3)


# ----------------------------------------------------------------- roles


class TestRoles:
    """Not every optimizer both provides and receives a scale.

    These no-ops are deliberate, and each is a place where a plausible
    "fix" would be wrong.
    """

    def test_lbfgs_provides_but_does_not_receive(self):
        read, write = build_transfer_fns("lbfgs", FAMILY_GRADIENT, {})
        state = _optax_stepped(optax.lbfgs(), steps=0)
        # Nothing has stepped yet, so there is no step to report.
        _, conf = read(state)
        assert float(conf) == CONF_ABSENT
        # After ``init`` every field is zeroed and the first step length
        # comes from a capped inverse gradient norm the line search then
        # overwrites -- there is nowhere for a scale to land.
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: bool(jnp.array_equal(a, b)),
                write(_bundle(), state),
                state,
            )
        )

    def test_de_provides_but_does_not_receive(self):
        # DE holds neither ``std`` nor ``velocity``, so its scale is the
        # population spread -- readable, but the population is the
        # caller's ``best_one`` handshake, not the writer's to set.
        _, state, read, write = _evosax_pop(
            DifferentialEvolution, "differentialevolution", generations=1
        )
        _, conf = read(state)
        assert float(conf) > CONF_ABSENT
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: bool(jnp.array_equal(a, b)),
                write(_bundle(), state),
                state,
            )
        )

    def test_rmsprop_receives_but_does_not_provide(self):
        read, write = build_transfer_fns(
            "rmsprop", FAMILY_GRADIENT, {"learning_rate": 1e-2}
        )
        state = _optax_stepped(optax.rmsprop(1e-2))
        # A preconditioner with no stored step cannot report a scale:
        # the step it would take depends on a gradient magnitude nothing
        # retained.
        _, conf = read(state)
        assert float(conf) == CONF_ABSENT
        # It can still be rescaled, using the unit-gradient reference.
        before = optax.tree.get(state, "nu")
        after = optax.tree.get(write(_bundle(), state), "nu")
        assert not bool(jnp.allclose(before, after))

    def test_randomsearch_neither(self):
        spec = transfer_spec_for("randomsearch", FAMILY_POPULATION)
        assert spec.reader == "none"
        assert spec.writer == "none"


# ------------------------------------------------------------- the gate


class TestAbsenceGate:
    """``CONF_ABSENT`` is the only level that changes behaviour."""

    @pytest.mark.parametrize(
        "cls", [CR_FM_NES, Sep_CMA_ES], ids=lambda c: c.__name__
    )
    def test_absent_bundle_leaves_scale_untouched(self, cls):
        _, state, _, write = _evosax_dist(cls)
        written = write(_bundle(conf=CONF_ABSENT), state)
        assert bool(jnp.allclose(written.std, state.std))

    def test_zero_velocity_reports_absent_not_zero(self):
        # PSO's velocity is zero until a generation runs. Writing a zero
        # scale into the next optimizer would freeze it, so absence is
        # the honest answer.
        _, state, read, _ = _evosax_pop(PSO, "pso", generations=0)
        sigma, conf = read(state)
        assert float(conf) == CONF_ABSENT
        assert float(sigma) == 0.0

    def test_nan_scale_reports_absent(self):
        _, state, read, _ = _evosax_dist(CR_FM_NES)
        poisoned = state.replace(std=jnp.asarray(jnp.nan))
        _, conf = read(poisoned)
        assert float(conf) == CONF_ABSENT


# -------------------------------------------------- shape preservation


class TestShapePreservation:
    """Scale goes stale; shape does not. A write rescales, never flattens."""

    def test_rprop_preserves_relative_structure(self):
        _, write = build_transfer_fns(
            "rprop", FAMILY_GRADIENT, {"learning_rate": 1e-3}
        )
        state = optax.rprop(1e-3).init(jnp.ones((1, DIM)))
        # A deliberately uneven ratchet: the relative pattern is the
        # information rprop has accumulated.
        uneven = jnp.linspace(1e-4, 1e-2, DIM).reshape(1, DIM)
        state = optax.tree.set(state, step_sizes=uneven)

        after = optax.tree.get(write(_bundle(), state), "step_sizes")
        ratios = after / uneven
        assert bool(jnp.allclose(ratios, ratios[0, 0], rtol=1e-5))
        rms = jnp.sqrt(jnp.mean(jnp.square(after)))
        assert jnp.isclose(rms, TARGET, rtol=1e-5)

    def test_rprop_zeroes_prev_updates(self):
        # ``sign(g * prev_updates)`` keys the ratchet; across an iterate
        # jump those signs describe a different neighbourhood. Zeroing
        # takes the branch that leaves ``step_sizes`` alone for one
        # clean step.
        _, write = build_transfer_fns(
            "rprop", FAMILY_GRADIENT, {"learning_rate": 1e-3}
        )
        state = _optax_stepped(optax.rprop(1e-3))
        assert not bool(
            jnp.allclose(optax.tree.get(state, "prev_updates"), 0.0)
        )
        after = write(_bundle(), state)
        assert bool(jnp.allclose(optax.tree.get(after, "prev_updates"), 0.0))

    def test_snes_per_coordinate_std_is_rescaled_not_flattened(self):
        _, state, _, write = _evosax_dist(SNES)
        uneven = jnp.linspace(0.5, 2.0, DIM)
        state = state.replace(std=uneven)
        after = write(_bundle(), state).std
        ratios = after / uneven
        assert bool(jnp.allclose(ratios, ratios[0], rtol=1e-5))

    def test_crfmnes_keeps_shape_and_zeroes_paths(self):
        # The whole reason own-ask beats a reset: ``D`` and ``v`` are the
        # landscape geometry and survive, while the cumulation paths are
        # calibrated against the algorithm's own past steps and must not.
        _, state, _, write = _evosax_dist(CR_FM_NES)
        state = state.replace(p_std=jnp.ones((DIM,)), p_c=jnp.ones((DIM,)))
        after = write(_bundle(), state)
        assert bool(jnp.allclose(after.D, state.D))
        assert bool(jnp.allclose(after.v, state.v))
        assert bool(jnp.allclose(after.p_std, 0.0))
        assert bool(jnp.allclose(after.p_c, 0.0))

    def test_distribution_write_sets_mean_to_handed_point(self):
        algo, state, _, write = _evosax_dist(Sep_CMA_ES)
        after = write(_bundle(), state)
        assert bool(
            jnp.allclose(after.mean, algo._ravel_solution(_bundle().x))
        )


# --------------------------------------------------- pytree-shaped params


class _TinyPytreeParams(eqx.Module):
    """Stand-in for a decision vector that is itself a pytree.

    Some tasks' "params" are not a flat array -- e.g.
    ``gaussian_classification``'s decision vector is literally the
    weights of a small classifier network (an ``eqx.Module``). Any
    optimizer-state field that mirrors ``params`` (Rprop's
    ``step_sizes``, L-BFGS's ``diff_params_memory``) then comes back
    shaped like this class too, not like a bare array.
    """

    w: jax.Array
    b: jax.Array
    activation: Callable = eqx.field(static=True, default=jax.nn.relu)


class TestPytreeParams:
    """Readers/writers must not assume a bare array under the hood.

    Regression coverage for a bug where ``gaussian_classification``'s
    MLP-shaped params made ``read_sigma_rprop`` / ``make_write_rprop`` /
    ``make_read_sigma_lbfgs`` raise ``TypeError: float() argument must
    be a string or a real number, not 'MLP'`` -- ``_rms`` and
    ``_rescale_to`` assumed every leaf they touched was already a bare
    array.
    """

    @staticmethod
    def _params():
        return _TinyPytreeParams(w=jnp.ones((3, 2)), b=jnp.zeros((2,)))

    def test_rprop_round_trips_pytree_params(self):
        params = self._params()
        read, write = build_transfer_fns(
            "rprop", FAMILY_GRADIENT, {"learning_rate": 1e-3}
        )
        state = optax.rprop(1e-3).init(params)
        sigma, conf = read(write(_bundle(), state))
        assert float(conf) > CONF_ABSENT
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    def test_lbfgs_reads_scale_off_pytree_params_after_steps(self):
        params = self._params()
        read, _ = build_transfer_fns("lbfgs", FAMILY_GRADIENT, {})
        solver = optax.lbfgs()
        state = solver.init(params)

        def loss(p):
            return sum(
                jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(p)
            )

        for _ in range(3):
            value, grads = jax.value_and_grad(loss)(params)
            updates, state = solver.update(
                grads, state, params, value=value, grad=grads, value_fn=loss
            )
            params = optax.apply_updates(params, updates)

        sigma, conf = read(state)
        assert float(conf) > CONF_ABSENT
        assert bool(jnp.isfinite(sigma))


# ------------------------------------------------- mr15ga's 1/5 rule


class TestMR15BaselineRepair:
    """The scale write must survive MR15-GA's next ``tell``.

    Its 1/5 rule reads ``mean(fitness < state.fitness)`` against the
    *stored elite*. A stale elite from an early generation makes a
    population drawn around a much better point look like near-total
    success, doubling the mutation scale exactly when it should shrink
    -- undoing the write one generation later. This is the decoder rule
    most likely to be wrong and least likely to be caught by asserting
    on the written state alone, because the failure happens afterwards.
    """

    def test_baseline_is_rebased_on_the_handed_over_best(self):
        _, state, _, write = _evosax_pop(MR15_GA, "mr15ga")
        after = write(_bundle(), state)
        assert bool(jnp.allclose(after.fitness, -1.0))
        assert bool(jnp.allclose(after.population, 0.5))

    def test_stale_elite_would_have_doubled_the_scale(self):
        algo, state, _, write = _evosax_pop(MR15_GA, "mr15ga")
        # A deliberately terrible stored elite, as if MR15-GA last ran
        # early in the episode.
        stale = state.replace(fitness=jnp.full((POPSIZE,), 1e6))
        after = write(_bundle(), stale)

        key = jr.key(1)
        # Every candidate beats the stale elite but not the handed-over
        # best, which is the situation the repair exists for.
        fitness = jnp.full((POPSIZE,), 0.0)
        params = algo.default_params

        repaired, _ = algo.tell(
            key, jnp.zeros((POPSIZE, DIM)), fitness, after, params
        )
        unrepaired, _ = algo.tell(
            key, jnp.zeros((POPSIZE, DIM)), fitness, stale, params
        )
        assert float(repaired.std) < float(after.std)
        assert float(unrepaired.std) > float(stale.std)


# ------------------------------------------------------ structural guards


class TestNonFiniteBundleIsRefused:
    """A poisoned bundle must not poison the receiver.

    ``sigma`` is already protected at the reader: every one routes
    through ``_finite_or_absent``, which downgrades a non-finite or
    non-positive scale to ``CONF_ABSENT``. ``x`` and ``f`` had no such
    guard, and the writes that consume them were ungated -- so this
    class covers the whole bundle, and asserts the *fallback*, not just
    the absence of NaN: the receiver keeps what it already had.
    """

    @pytest.fixture(scope="class")
    def problem(self):
        return sphere_problem(DIM)

    @staticmethod
    def _built(problem, name, hp):
        update_class = REGISTRY[name](
            **problem, opt_hash=0, bounded=(None, None), stop_fn=None, **hp
        )
        params = jnp.zeros((update_class.popsize, DIM))
        return update_class, update_class.init_fn(params, jr.key(0))

    @staticmethod
    def _newly_non_finite(before, after):
        """Leaves finite before the write and non-finite after it.

        Compared per leaf rather than globally because several states
        carry legitimate sentinels at ``init`` -- L-BFGS's
        ``value = inf``, the evosax algorithms' ``best_solution = NaN``
        -- which a whole-tree finiteness check would flag as damage.
        """
        flat = jax.tree_util.tree_flatten_with_path
        pb = {jax.tree_util.keystr(k): v for k, v in flat(before)[0]}
        pa = {jax.tree_util.keystr(k): v for k, v in flat(after)[0]}
        bad = []
        for key, value in pb.items():
            arr = jnp.asarray(value)
            if not jnp.issubdtype(arr.dtype, jnp.inexact):
                continue
            was = bool(jnp.all(jnp.isfinite(arr.astype(float))))
            now = bool(jnp.all(jnp.isfinite(jnp.asarray(pa[key], float))))
            if was and not now:
                bad.append(key)
        return bad

    POISON = [
        ("sigma_nan", jnp.zeros((1, DIM)), 0.0, jnp.nan),
        ("sigma_inf", jnp.zeros((1, DIM)), 0.0, jnp.inf),
        ("x_nan", jnp.full((1, DIM), jnp.nan), 0.0, 0.05),
        ("x_inf", jnp.full((1, DIM), jnp.inf), 0.0, 0.05),
        ("f_nan", jnp.zeros((1, DIM)), jnp.nan, 0.05),
        ("f_inf", jnp.zeros((1, DIM)), jnp.inf, 0.05),
    ]

    @pytest.mark.parametrize(
        ("label", "x", "f", "sigma"), POISON, ids=[c[0] for c in POISON]
    )
    @pytest.mark.parametrize("name", list(MENU))
    def test_no_leaf_is_poisoned(self, problem, name, label, x, f, sigma):
        update_class, opt_state = self._built(problem, name, MENU[name])
        if update_class.transfer_write_fn is None:
            pytest.skip(f"{name} has no writer")
        # ``conf_sigma`` is deliberately confident: the point is that a
        # writer must not take a caller's word for it.
        bundle = TransferBundle(
            x=x,
            f=jnp.asarray(f, dtype=float),
            sigma=jnp.asarray(sigma, dtype=float),
            conf_sigma=jnp.asarray(CONF_EXACT, dtype=float),
        )
        after = update_class.transfer_write_fn(bundle, opt_state)
        assert not self._newly_non_finite(opt_state, after), (
            f"{name} poisoned by {label}"
        )

    def test_distribution_keeps_its_own_mean(self, problem):
        """The fallback, stated positively.

        A NaN ``mean`` is unrecoverable -- every ``ask`` returns NaN,
        every loss is NaN, and no ``tell`` can undo it, so the optimizer
        is dead for the rest of the episode. Keeping its own mean lets it
        carry on from where it was.
        """
        update_class, opt_state = self._built(problem, "crfmnes", {})
        bundle = TransferBundle(
            x=jnp.full((1, DIM), jnp.nan),
            f=jnp.asarray(0.0),
            sigma=jnp.asarray(0.05),
            conf_sigma=jnp.asarray(CONF_EXACT),
        )
        after = update_class.transfer_write_fn(bundle, opt_state)
        assert jnp.array_equal(after.mean, opt_state.mean)
        # The usable half still lands -- one bad field does not veto the
        # other.
        assert float(after.std) != float(opt_state.std)
        population, _ = update_class.ask_fn(after, jr.key(1))
        leaves = jax.tree.leaves(population)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)

    def test_mr15ga_keeps_its_elite_when_f_is_infinite(self, problem):
        """``+inf`` is the reachable case, and it inverts the 1/5 rule.

        ``best_loss`` starts at ``+inf`` and stays there while nothing
        has improved -- an entirely-NaN first generation, which this
        ecosystem sees at high ambient dimension. Rebasing the stored
        elite onto ``+inf`` makes every candidate beat it, so the 1/5
        rule reads a 100% success rate and doubles the width right after
        the scale write set it.
        """
        update_class, opt_state = self._built(problem, "mr15ga", {})
        opt_state = opt_state.replace(
            fitness=jnp.full_like(jnp.asarray(opt_state.fitness), 100.0)
        )
        bundle = TransferBundle(
            x=jnp.zeros((1, DIM)),
            f=jnp.asarray(jnp.inf),
            sigma=jnp.asarray(0.05),
            conf_sigma=jnp.asarray(CONF_EXACT),
        )
        after = update_class.transfer_write_fn(bundle, opt_state)
        assert jnp.array_equal(after.fitness, opt_state.fitness)
        assert jnp.array_equal(after.population, opt_state.population)
        # And a finite ``f`` still rebases, so the gate is not blanket.
        ok = update_class.transfer_write_fn(
            TransferBundle(
                x=jnp.zeros((1, DIM)),
                f=jnp.asarray(1.0),
                sigma=jnp.asarray(0.05),
                conf_sigma=jnp.asarray(CONF_EXACT),
            ),
            opt_state,
        )
        assert jnp.allclose(ok.fitness, 1.0)


class TestRegistryGuards:
    """Guards over the whole registry, not one optimizer at a time."""

    @pytest.fixture(scope="class")
    def problem(self):
        return sphere_problem(DIM)

    def test_menu_builds_with_transfer_wired(self, problem):
        for i, (name, hp) in enumerate(MENU.items()):
            update_class = REGISTRY[name](
                **problem,
                opt_hash=i,
                bounded=(None, None),
                stop_fn=None,
                **hp,
            )
            assert update_class.family in FAMILIES, name
            assert callable(update_class.transfer_read_fn), name
            assert callable(update_class.transfer_write_fn), name

    def test_own_ask_optimizers_expose_ask(self, problem):
        # A missing ``ask_fn`` does not raise -- it silently degrades to
        # the old population-substitution path, which is exactly the bug
        # this work exists to remove. So it gets its own guard.
        #
        # The condition is ``TransferSpec.own_ask``, not the family:
        # every ``distribution`` algorithm needs its own ask, and so do
        # the mutation GAs, whose next generation is scored against the
        # baseline the writer just repaired (``docs/adr/0014``). A spec
        # that claims ``own_ask`` while its factory forgets ``ask_fn``
        # is the silent-degradation case, and vice versa is an ask the
        # caller will never reach.
        for i, (name, hp) in enumerate(MENU.items()):
            update_class = REGISTRY[name](
                **problem,
                opt_hash=i,
                bounded=(None, None),
                stop_fn=None,
                **hp,
            )
            spec = transfer_spec_for(name, update_class.family)
            if update_class.family == FAMILY_DISTRIBUTION:
                assert spec.own_ask, name
            if spec.own_ask:
                assert update_class.ask_fn is not None, name
            else:
                assert update_class.ask_fn is None, name

    def test_registry_binds_the_resolved_spec(self, problem):
        """A factory that forgets ``name=`` must fail here, loudly.

        ``build_transfer_fns`` keys the ``TRANSFER_OVERRIDES`` lookup on
        the registry name, and every generic classmethod defaults
        ``name`` to ``""``. A factory that calls one of them without
        passing its own name therefore resolves to the *family default*
        instead of its override -- and that degrades silently in the
        worst way: the family reader finds no field it recognises and
        returns ``CONF_ABSENT``, so the optimizer simply stops carrying a
        scale. Nothing raises, no population is malformed, and the only
        symptom is a switch that quietly transfers nothing.

        This is not hypothetical. ``lbfgs_update`` delegates to
        ``optax_update_extra_kwargs`` on deterministic tasks
        (``not problem["pass_rng"]``, which is most of them) and shipped
        without ``name="lbfgs"``, so L-BFGS reported no scale on every
        switch out of it. The tests then in place all passed: the
        provider assertions are gated on the bundle carrying something,
        and the "``lbfgs`` receives nothing" assertion held because the
        wrongly-bound *gradient* writer also happened to no-op on a state
        with no second-moment field. Identity of the bound function is
        the only check that catches it.
        """
        for i, name in enumerate(TRANSFER_OVERRIDES):
            if name not in REGISTRY:
                continue
            update_class = REGISTRY[name](
                **problem,
                opt_hash=i,
                bounded=(None, None),
                stop_fn=None,
                **MENU.get(name, {}),
            )
            spec = transfer_spec_for(name, update_class.family)
            expected_read, expected_write = build_transfer_fns(
                name,
                update_class.family,
                MENU.get(name, {}),
                ravel_fn=None,
            )
            assert update_class.family == spec.family, name
            # Compare by ``__name__``: the builders return closures for
            # the parameterised cases, so identity holds only for the
            # module-level readers/writers. ``None`` is a legitimate
            # binding -- ``randomsearch`` holds no state, and both
            # consumers treat ``None`` as "no information" -- but only
            # where the spec says so, never as the result of a forgotten
            # ``name=``, which binds a real family-default function.
            if update_class.transfer_read_fn is None:
                assert spec.reader == "none", (
                    f"{name}: no reader bound, spec says {spec.reader}"
                )
            else:
                assert (
                    update_class.transfer_read_fn.__name__
                    == expected_read.__name__
                ), f"{name}: bound the wrong reader ({spec.reader} expected)"
            if update_class.transfer_write_fn is None:
                assert spec.writer == "none", (
                    f"{name}: no writer bound, spec says {spec.writer}"
                )
            else:
                assert (
                    update_class.transfer_write_fn.__name__
                    == expected_write.__name__
                ), f"{name}: bound the wrong writer ({spec.writer} expected)"

    def test_lbfgs_reports_a_scale_on_a_deterministic_task(self, problem):
        """The regression above, pinned end to end.

        ``problem["pass_rng"]`` is false here, which is the path that was
        broken. After a few steps L-BFGS has accepted steps in
        ``diff_params_memory``, so its reader must report one --
        ``CONF_ABSENT`` here means the override is not bound.
        """
        assert not problem["pass_rng"]
        update_class = REGISTRY["lbfgs"](
            **problem, opt_hash=0, bounded=(None, None), stop_fn=None
        )
        params = jnp.zeros((update_class.popsize, DIM))
        opt_state = update_class.init_fn(params, jr.key(0))

        # Before any accepted step the memory is zeros: correctly absent.
        _, conf_at_init = update_class.transfer_read_fn(opt_state)
        assert float(conf_at_init) == CONF_ABSENT

        carry = (params, opt_state, jr.key(0))
        for _ in range(3):
            carry, _ = update_class.step_fn(carry, {})
        sigma, conf = update_class.transfer_read_fn(carry[1])
        assert float(conf) == CONF_EXACT
        assert float(sigma) > 0.0
        assert jnp.isfinite(sigma)

    def test_ask_fn_emits_a_full_population(self, problem):
        update_class = REGISTRY["crfmnes"](
            **problem, opt_hash=0, bounded=(None, None), stop_fn=None
        )
        params = jnp.zeros((update_class.popsize, DIM))
        opt_state = update_class.init_fn(params, jr.key(0))
        population, _ = update_class.ask_fn(opt_state, jr.key(1))
        leaves = jax.tree.leaves(population)
        assert leaves
        assert all(leaf.shape[0] == update_class.popsize for leaf in leaves)

    def test_readers_agree_on_output_layout(self, problem):
        """All readers must be usable in one ``lax.switch``.

        The bundle crosses a switch on the outgoing optimizer's index,
        so every branch has to return the same structure -- two scalar
        floats. A reader returning a differently shaped or typed value
        would only fail at trace time, deep inside a rollout.
        """
        built = [
            REGISTRY[name](
                **problem,
                opt_hash=i,
                bounded=(None, None),
                stop_fn=None,
                **hp,
            )
            for i, (name, hp) in enumerate(MENU.items())
        ]
        states = [
            uc.init_fn(jnp.zeros((uc.popsize, DIM)), jr.key(0)) for uc in built
        ]

        def dispatch(index):
            return jax.lax.switch(
                index,
                tuple(
                    (lambda _, uc=uc, st=st: uc.transfer_read_fn(st))
                    for uc, st in zip(built, states, strict=True)
                ),
                0,
            )

        for index in range(len(built)):
            sigma, conf = jax.jit(dispatch)(index)
            assert jnp.shape(sigma) == ()
            assert jnp.shape(conf) == ()
            assert sigma.dtype == conf.dtype

    def test_lbfgs_reports_its_very_first_step(self, problem):
        """One iteration is a visit length the policy actually picks.

        ``lbfgs`` is the one optimizer tabulated ``reset``, so its
        ``count`` restarts on every switch into it, and "run lbfgs for 1
        iteration" is one of the four budget choices. The memory holds
        no *difference* yet at that point, but the step itself is
        reconstructible -- and the reconstruction has to match the step
        that was actually taken, not merely be finite.
        """
        update_class = REGISTRY["lbfgs"](
            **problem, opt_hash=0, bounded=(None, None), stop_fn=None
        )
        before = jnp.zeros((update_class.popsize, DIM))
        carry = (before, update_class.init_fn(before, jr.key(0)), jr.key(0))

        carry, _ = update_class.step_fn(carry, {})
        after, opt_state, _ = carry

        sigma, conf = update_class.transfer_read_fn(opt_state)
        assert float(conf) == CONF_EXACT
        taken = jnp.sqrt(jnp.mean(jnp.square(after[0] - before[0])))
        assert float(taken) > 0.0
        assert jnp.isclose(sigma, taken, rtol=1e-4)


class TestOwnAsk:
    """Which optimizers draw their own post-switch population.

    A per-optimizer property, not a family one (``docs/adr/0014``). The
    cost of getting it wrong runs both ways: an optimizer that needs its
    own ask and does not get one is scored against a baseline it was not
    drawn from, and one that gets an ask it cannot use degenerates.
    """

    @pytest.mark.parametrize(
        "name", ["crfmnes", "sepcmaes", "cmaes", "snes", "mr15ga"]
    )
    def test_ask_coupled_and_mutation_gas_opt_in(self, name):
        family = FAMILY_POPULATION if name == "mr15ga" else FAMILY_DISTRIBUTION
        assert transfer_spec_for(name, family).own_ask

    @pytest.mark.parametrize("name", ["samrga", "gesmrga"])
    def test_the_other_mutation_gas_opt_in(self, name):
        spec = transfer_spec_for(name, FAMILY_POPULATION)
        assert spec.own_ask
        # The repairing writer and the own ask travel together: the ask
        # is only meaningful against a repaired baseline, and a repaired
        # baseline is only honest if the population is drawn from it.
        assert spec.writer == "population_repair"

    @pytest.mark.parametrize(
        "name", ["differentialevolution", "shade", "pso", "randomsearch"]
    )
    def test_difference_and_velocity_methods_stay_out(self, name):
        # DE and SHADE build mutants from difference vectors of the
        # stored population; the repair makes every row identical, and
        # identical rows have no differences. PSO's ask reads
        # ``population_best`` / ``fitness_best``, which no writer
        # repairs. Randomsearch holds nothing at all.
        assert not transfer_spec_for(name, FAMILY_POPULATION).own_ask

    def test_simple_ga_is_excluded_because_its_std_expires(self):
        """A deliberate exclusion, pinned so a "fix" has to argue.

        ``SimpleGA`` is a mutation GA of the same shape as the three
        that opt in, but its ``_tell`` overwrites ``std`` from a fixed
        schedule, so a transferred scale does not survive one
        generation. Giving it the repairing writer would create a write
        whose effect silently expires.
        """
        spec = transfer_spec_for("simplega", FAMILY_POPULATION)
        assert not spec.own_ask
        assert spec.writer == "population"

        algo, state, _, write = _evosax_pop(SimpleGA, "simplega")
        written = write(_bundle(), state)
        assert jnp.isclose(written.std, TARGET, rtol=1e-5)

        key = jr.key(0)
        population, asked = algo.ask(key, written, algo.default_params)
        told, _ = algo.tell(
            key,
            population,
            jnp.arange(POPSIZE, dtype=float),
            asked,
            algo.default_params,
        )
        # One generation later the schedule has taken it back.
        assert not jnp.isclose(told.std, TARGET, rtol=1e-5)


class TestBaselineRepair:
    """The rebase is unconditional on anything about the problem.

    Writing ``fitness = bundle.f`` claims a measurement: that
    re-evaluating ``bundle.x`` would return it. On a stochastic loss it
    would not, since ``best_loss`` is a running minimum and so the
    luckiest draw. The claim is made anyway.

    An earlier version gated this on ``problem["pass_rng"]``. That is not a
    property a handshake may read: a real problem does not come
    labelled, the caller usually cannot say, and an update rule whose
    shape depends on how the objective was declared is not one you can
    reason about. The bias the claim adds is also the bias MR15-GA
    already has -- its ``(µ+λ)`` selection keeps the better of each
    pair, so its own stored fitness is a running minimum too, and its
    scale collapses under noise with no switch at all. Refusing the
    claim would not fix that, and with own-ask it is the only way the
    handed-over *location* reaches a mutation GA at all.
    """

    def _mr15ga(self):
        algo = MR15_GA(population_size=POPSIZE, solution=jnp.zeros((DIM,)))
        state = algo.init(
            jr.key(0),
            jr.normal(jr.key(0), (POPSIZE, DIM)),
            jnp.arange(POPSIZE, dtype=float),
            algo.default_params,
        )
        _, write = build_transfer_fns("mr15ga", FAMILY_POPULATION, {})
        return algo, state, write

    def test_the_elite_is_rebased_on_the_handed_over_best(self):
        _, state, write = self._mr15ga()
        written = write(_bundle(), state)
        assert jnp.allclose(written.population, 0.5)
        assert jnp.allclose(written.fitness, -1.0)
        assert jnp.isclose(written.std, TARGET, rtol=1e-5)

    def test_the_writer_reads_nothing_about_the_task(self):
        """The guard against the gate coming back.

        ``build_transfer_fns`` takes the optimizer's name, family and
        hyperparameters -- nothing that could tell it whether the loss
        is stochastic. Two writers built the same way must behave
        identically, whatever the task they end up attached to.
        """
        _, state, one = self._mr15ga()
        _, _, two = self._mr15ga()
        first, second = one(_bundle(), state), two(_bundle(), state)
        assert jnp.array_equal(first.population, second.population)
        assert jnp.array_equal(first.fitness, second.fitness)
        assert jnp.array_equal(first.std, second.std)

    def test_an_unusable_f_is_still_refused(self):
        """A property of the numbers in hand, not of the problem.

        ``+inf`` is the opposite failure to a lucky minimum: every
        candidate beats it, so the 1/5 rule reads a 100% success rate
        and doubles the width immediately after the scale write set it.
        ``best_loss`` is ``+inf`` until something improves, which at
        high ambient dimension can be a whole first generation of NaN.
        """
        _, state, write = self._mr15ga()
        written = write(
            TransferBundle(
                x=jnp.full((DIM,), 0.5),
                f=jnp.asarray(jnp.inf),
                sigma=jnp.asarray(TARGET),
                conf_sigma=jnp.asarray(CONF_EXACT),
            ),
            state,
        )
        assert jnp.array_equal(written.fitness, state.fitness)
        assert jnp.array_equal(written.population, state.population)
        # The scale is a separate judgement and still lands.
        assert jnp.isclose(written.std, TARGET, rtol=1e-5)


class TestMovesIterate:
    """The writes that only make sense if the iterate actually moved.

    A handshake does two different things at once. It refreshes a scale
    that has gone stale, and it repairs the damage of jumping the
    iterate somewhere else — re-centring a ``mean``, zeroing a
    cumulation path, rebasing an elite, clearing Rprop's
    ``prev_updates``. The second half is only justified by the jump. A
    caller that knows nothing moved (the incoming optimizer already held
    the best point) says so with ``moves_iterate=False``, and then those
    writes are not conservative but destructive: they discard adaptation
    the incoming optimizer earned, to protect it from a jump that is not
    happening.

    The scale write is deliberately *not* gated: an outgoing optimizer
    that found nothing better may still have learned the local scale.
    """

    def _still(self, sigma: float = TARGET):
        """A bundle that carries a scale but moves nothing."""
        return TransferBundle(
            x=jnp.full((DIM,), 0.5),
            f=jnp.asarray(-1.0),
            sigma=jnp.asarray(sigma),
            conf_sigma=jnp.asarray(CONF_EXACT),
            moves_iterate=jnp.asarray(False),
        )

    def test_defaults_to_moving(self):
        # The conservative default: a caller that does not know moves
        # the iterate, which is what every write is written for.
        assert bool(_bundle().moves_iterate)
        assert bool(
            TransferBundle.absent(jnp.zeros((DIM,)), 1.0).moves_iterate
        )

    def test_distribution_keeps_its_mean_and_paths(self):
        _, state, read, write = _evosax_dist(CR_FM_NES)
        stepped = state.replace(
            mean=jnp.full((DIM,), -0.25),
            p_std=jnp.full((DIM,), 0.3),
            p_c=jnp.full((DIM,), 0.4),
        )
        written = write(self._still(), stepped)

        assert jnp.allclose(written.mean, stepped.mean)
        assert jnp.allclose(written.p_std, stepped.p_std)
        assert jnp.allclose(written.p_c, stepped.p_c)

        # ...but the scale still lands.
        sigma, conf = read(written)
        assert float(conf) > CONF_ABSENT
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    def test_mutation_ga_keeps_its_elite(self):
        _, state, read, write = _evosax_pop(MR15_GA, "mr15ga")
        written = write(self._still(), state)

        assert jnp.array_equal(written.population, state.population)
        assert jnp.array_equal(written.fitness, state.fitness)
        sigma, _ = read(written)
        assert jnp.isclose(sigma, TARGET, rtol=1e-5)

    def test_rprop_keeps_its_previous_step(self):
        read, write = build_transfer_fns(
            "rprop", FAMILY_GRADIENT, {"learning_rate": 1e-3}
        )
        state = _optax_stepped(optax.rprop(1e-3))
        before = optax.tree.get(state, "prev_updates")
        assert not jnp.allclose(before, 0.0), "need a non-zero buffer"

        written = write(self._still(), state)

        # Rprop's ratchet keys on ``sign(g * prev_updates)``; those signs
        # describe a *different neighbourhood* only if the iterate
        # jumped. Keeping them also spares the dead iteration optax's
        # one-generation lag would otherwise cost.
        assert jnp.allclose(optax.tree.get(written, "prev_updates"), before)
        sigma, _ = read(written)
        assert jnp.isclose(sigma, TARGET, rtol=1e-3)

    def test_a_non_finite_point_never_moves_the_iterate(self):
        """The two gates compose, and the finiteness one still wins."""
        _, state, _, write = _evosax_dist(CR_FM_NES)
        stepped = state.replace(mean=jnp.full((DIM,), -0.25))
        poisoned = TransferBundle(
            x=jnp.full((DIM,), jnp.nan),
            f=jnp.asarray(-1.0),
            sigma=jnp.asarray(TARGET),
            conf_sigma=jnp.asarray(CONF_EXACT),
            moves_iterate=jnp.asarray(True),
        )
        assert jnp.allclose(write(poisoned, stepped).mean, stepped.mean)


class TestTransferSpec:
    """The override table, and that it stays small."""

    def test_family_default_when_not_overridden(self):
        spec = transfer_spec_for("cmaes", FAMILY_DISTRIBUTION)
        assert spec.family == FAMILY_DISTRIBUTION
        assert spec.reader == "distribution"

    @pytest.mark.parametrize(
        ("name", "family"),
        [
            ("rprop", FAMILY_GRADIENT),
            ("lbfgs", FAMILY_GRADIENT),
            ("mr15ga", FAMILY_POPULATION),
            ("randomsearch", FAMILY_POPULATION),
        ],
    )
    def test_overrides_win(self, name, family):
        assert transfer_spec_for(name, family) is not transfer_spec_for(
            "cmaes", FAMILY_DISTRIBUTION
        )

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match="family must be one of"):
            transfer_spec_for("cmaes", "not-a-family")

    def test_distribution_writer_requires_ravel_fn(self):
        # evosax stores ``mean`` raveled, so a pytree ``x`` cannot be
        # written without the algorithm's own flattening function.
        with pytest.raises(ValueError, match="ravel_fn"):
            build_transfer_fns("cmaes", FAMILY_DISTRIBUTION, {})
