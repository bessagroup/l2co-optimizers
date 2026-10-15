"""What every switchable optimizer declares to a switching loop (ADR 0012).

At a switch, l2co decides whether the incoming optimizer's own state
still holds where it is sent (l2co ADR 0022). For that every switchable
optimizer has to say where it is, what shape it has learned and how fast
that shape adapts, and what a restart keeps. These tests pin:

* the **contract**: every switchable registry entry declares all of it,
  identically on both construction paths, and an optimizer with a shape
  reader always has evidence rates (no rate, no reader);
* the **readers**: each turns its own fields into the one unit -- a
  determinant-1 shape plus the geometric-mean width it pairs with --
  with the right power, checked against a covariance built by hand;
* the **distance** in a shape's own widths: along a direction the shape
  says is wide an offset counts for less, across it for more;
* the **restart**: the trust-region optimizers keep their archive and
  nothing else.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# Local
from l2co_optimizers import (
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    PLAIN_RUN_OPTIMIZERS,
    OptimizationStep,
    Shape,
    build_switch_ports,
    optimizer_parts,
)
from l2co_optimizers import optimizers as REGISTRY
from l2co_optimizers._src.evosax_implementations import normalized_evosax

from .toy_problems import sphere_problem

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

DIM = 6
SWITCHABLE = sorted(set(REGISTRY) - set(PLAIN_RUN_OPTIMIZERS))

#: Every distribution ES that reads its own shape, with the evosax
#: parameters its evidence rates sum (diagonal, direction).
SHAPED = {
    "crfmnes": (("lr_B",), ("c1",)),
    "cmaes": (("c_1", "c_mu"), ("c_1", "c_mu")),
    "sepcmaes": (("c_1", "c_mu"), ()),
    "maes": (("c_1", "c_mu"), ("c_1", "c_mu")),
    "xnes": (("lr_B",), ("lr_B",)),
    "snes": (("lr_std_init",), ()),
    "simplees": (("c_std",), ()),
    "iamalgamfull": (("eta_std",), ("eta_std",)),
    "iamalgamunivariate": (("eta_std",), ()),
}


@pytest.fixture(scope="module")
def problem():
    return sphere_problem(DIM)


def _built(problem, name):
    """The registry entry and its parts, built from the same step."""
    for hyperparameters in ({}, {"learning_rate": 1e-2}):
        try:
            update_class = REGISTRY[name](
                **problem,
                opt_hash=0,
                bounded=(None, None),
                stop_fn=None,
                **hyperparameters,
            )
            parts = optimizer_parts(
                OptimizationStep(
                    optimizer=name, hyperparameters=hyperparameters
                ),
                model=problem["model"],
                loss_fn=problem["loss_fn"],
                pass_rng=problem["pass_rng"],
            )
            return update_class, parts
        except TypeError:
            continue
    pytest.skip(f"{name} cannot be built with or without a learning rate")


def _log_det(shape: Shape):
    return float(
        2.0 * jnp.sum(jnp.log(shape.diagonal))
        + jnp.sum(jnp.log(shape.stretches))
    )


def _es_state(name, **fields):
    """A fresh evosax state for ``name`` with ``fields`` replaced."""
    algo = normalized_evosax[name](population_size=8, solution=jnp.zeros(DIM))
    params = algo.default_params
    state = algo.init(jr.key(0), jnp.zeros(DIM), params)
    return state.replace(**fields), params


def _ports(name):
    _, params = _es_state(name)
    return build_switch_ports(name, FAMILY_DISTRIBUTION, params)


class TestContract:
    @pytest.mark.parametrize("name", SWITCHABLE)
    def test_every_switchable_entry_declares_its_ports(self, problem, name):
        update_class, parts = _built(problem, name)
        ports = update_class.switch_ports
        assert ports is not None, f"{name}: no switch ports"
        assert parts.switch_ports is not None, f"{name}: parts lack ports"
        # Both construction paths declare the same thing.
        other = parts.switch_ports
        assert ports.read_point is other.read_point, name
        assert ports.read_shape is other.read_shape, name
        assert ports.shape_rates == other.shape_rates, name
        assert ports.restart_keeps == other.restart_keeps, name

    @pytest.mark.parametrize("name", SWITCHABLE)
    def test_a_shape_reader_comes_with_rates(self, problem, name):
        ports = _built(problem, name)[0].switch_ports
        if ports.read_shape is None:
            assert ports.shape_rates is None, name
            return
        diagonal_rate, direction_rate = ports.shape_rates
        assert diagonal_rate > 0.0, name
        assert direction_rate >= 0.0, name

    @pytest.mark.parametrize("name", SWITCHABLE)
    def test_the_point_is_declared(self, problem, name):
        update_class, _ = _built(problem, name)
        ports = update_class.switch_ports
        if update_class.family == FAMILY_GRADIENT or name == "randomsearch":
            # The iterate is the switching loop's; random search has none.
            assert ports.read_point is None, name
            return
        state = update_class.init_fn(
            jnp.zeros((update_class.popsize, DIM)), jr.key(0)
        )
        point = ports.read_point(state)
        assert point.shape == (DIM,), name
        assert bool(jnp.all(jnp.isfinite(point))), name

    @pytest.mark.parametrize("name", sorted(SHAPED))
    def test_shaped_optimizers_read_a_normalised_shape(self, problem, name):
        update_class, _ = _built(problem, name)
        ports = update_class.switch_ports
        assert ports.read_shape is not None, name
        state = update_class.init_fn(
            jnp.zeros((update_class.popsize, DIM)), jr.key(0)
        )
        shape = ports.read_shape(state)
        assert abs(_log_det(shape)) < 1e-5, name
        assert float(shape.scale) > 0.0, name

    def test_no_rate_means_no_reader(self):
        # A shape reader without es_params to read rates from is refused
        # rather than built with a guessed rate.
        with pytest.raises(ValueError, match="evidence rates"):
            build_switch_ports("cmaes", FAMILY_DISTRIBUTION, None)


class TestRates:
    @pytest.mark.parametrize("name", sorted(SHAPED))
    def test_rates_sum_the_declared_parameters(self, name):
        _, params = _es_state(name)
        diagonal_fields, direction_fields = SHAPED[name]
        ports = build_switch_ports(name, FAMILY_DISTRIBUTION, params)
        expected = (
            sum(float(getattr(params, f)) for f in diagonal_fields),
            sum(float(getattr(params, f)) for f in direction_fields),
        )
        np.testing.assert_allclose(ports.shape_rates, expected)

    def test_cma_counts_both_terms(self):
        # The covariance forgets at (1 - c_1 - c_mu) per generation;
        # c_1 alone understates the rate by about mu_eff.
        _, params = _es_state("cmaes")
        rate = _ports("cmaes").shape_rates[0]
        assert rate > float(params.c_1)


def _rotation(seed=0):
    q, _ = jnp.linalg.qr(jr.normal(jr.key(seed), (DIM, DIM)))
    return q


class TestReaders:
    def test_cmaes_finds_the_widest_and_narrowest_axes(self):
        q = _rotation()
        eig = jnp.array([100.0, 1.0, 1.0, 1.0, 1.0, 0.01])
        C = (q * eig) @ q.T
        state, _ = _es_state("cmaes", C=C, std=jnp.asarray(0.5))
        shape = _ports("cmaes").read_shape(state)
        assert bool(shape.has_direction[0]) and bool(shape.has_direction[1])
        # Directions live in the diagonal-scaled space. A wide one maps
        # back through D (the metric's large eigenvalue), a narrow one
        # through D⁻¹ (the inverse metric's large eigenvalue).
        widest = shape.diagonal * shape.directions[0]
        narrowest = shape.directions[1] / shape.diagonal
        cos_w = jnp.abs(widest @ q[:, 0]) / jnp.linalg.norm(widest)
        cos_n = jnp.abs(narrowest @ q[:, -1]) / jnp.linalg.norm(narrowest)
        assert float(cos_w) > 0.95
        assert float(cos_n) > 0.95
        assert float(shape.stretches[0]) > 1.0 > float(shape.stretches[1])
        assert abs(_log_det(shape)) < 1e-5

    def test_scale_is_the_geometric_mean_width(self):
        # std² C with det(C) = 1 has geometric-mean width std, however
        # stretched C is -- the RMS width would be far larger here.
        eig = jnp.array([1e4, 1.0, 1.0, 1.0, 1.0, 1e-4])
        state, _ = _es_state("cmaes", C=jnp.diag(eig), std=jnp.asarray(0.3))
        shape = _ports("cmaes").read_shape(state)
        np.testing.assert_allclose(float(shape.scale), 0.3, rtol=1e-6)

    def test_sepcmaes_reads_variances_not_widths(self):
        C = jnp.array([4.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        state, _ = _es_state("sepcmaes", C=C, std=jnp.asarray(1.0))
        shape = _ports("sepcmaes").read_shape(state)
        # C holds variances: a width ratio of 2, not 4.
        np.testing.assert_allclose(
            float(shape.diagonal[0] / shape.diagonal[1]), 2.0, rtol=1e-6
        )

    def test_crfmnes_is_native(self):
        D = jnp.array([1.0, 2.0, 1.0, 1.0, 0.5, 1.0])
        v = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        state, _ = _es_state("crfmnes", D=D, v=v, std=jnp.asarray(1.0))
        shape = _ports("crfmnes").read_shape(state)
        np.testing.assert_allclose(
            np.asarray(shape.directions[0]), np.asarray(v / jnp.sqrt(2.0))
        )
        np.testing.assert_allclose(float(shape.stretches[0]), 3.0)
        # Its v only widens, so it never reports a narrow direction.
        assert not bool(shape.has_direction[1])
        np.testing.assert_allclose(
            np.asarray(shape.diagonal / shape.diagonal[0]), np.asarray(D)
        )

    def test_xnes_reads_bt_b(self):
        # xNES samples z @ B, so its covariance is Bᵀ B, not B Bᵀ.
        B = jnp.eye(DIM).at[0, 1].set(3.0)
        state, _ = _es_state("xnes", B=B, std=jnp.asarray(1.0))
        shape = _ports("xnes").read_shape(state)
        cov = shape.scale**2 * (
            shape.diagonal[:, None] * shape.diagonal[None, :]
        )
        # Bᵀ B puts the extra variance on coordinate 1, B Bᵀ on 0.
        assert float(cov[1, 1]) > float(cov[0, 0])

    def test_iamalgam_adds_the_isotropic_part(self):
        state, _ = _es_state(
            "iamalgamfull", C=jnp.zeros((DIM, DIM)), std=jnp.asarray(2.0)
        )
        shape = _ports("iamalgamfull").read_shape(state)
        np.testing.assert_allclose(float(shape.scale), 2.0, rtol=1e-6)


class TestDistance:
    def test_neutral_distance_is_rms_over_scale(self):
        shape = Shape.neutral(DIM, jnp.asarray(0.5))
        offset = jnp.arange(1.0, DIM + 1.0)
        expected = float(jnp.sqrt(jnp.mean(offset**2)) / 0.5)
        np.testing.assert_allclose(float(shape.distance(offset)), expected)

    def test_along_a_wide_direction_is_nearer_than_across(self):
        eig = jnp.array([100.0, 1.0, 1.0, 1.0, 1.0, 0.01])
        state, _ = _es_state("cmaes", C=jnp.diag(eig), std=jnp.asarray(1.0))
        shape = _ports("cmaes").read_shape(state)
        along = jnp.zeros(DIM).at[0].set(1.0)
        across = jnp.zeros(DIM).at[5].set(1.0)
        assert float(shape.distance(along)) < float(shape.distance(across))
        # By the ratio of the widths along the two axes, sqrt(1e4).
        np.testing.assert_allclose(
            float(shape.distance(across) / shape.distance(along)),
            100.0,
            rtol=1e-3,
        )


class TestRestart:
    def test_default_restart_is_the_fresh_state(self, problem):
        update_class, _ = _built(problem, "cmaes")
        old = {"a": 1}
        fresh = {"a": 2}
        assert update_class.switch_ports.restart(old, fresh) is fresh

    @pytest.mark.parametrize("name", ["turbo", "rbftrustregion"])
    def test_trust_regions_keep_their_archive(self, problem, name):
        update_class, _ = _built(problem, name)
        ports = update_class.switch_ports
        assert ports.restart_keeps == (
            "archive",
            "archive_fitness",
            "archive_index",
        )
        params = jnp.zeros((update_class.popsize, DIM))
        old = update_class.init_fn(params, jr.key(0))
        old = old.replace(
            archive=jnp.ones_like(old.archive),
            archive_fitness=jnp.zeros_like(old.archive_fitness),
            archive_index=jnp.asarray(3, old.archive_index.dtype),
        )
        fresh = update_class.init_fn(params + 1.0, jr.key(1))
        restarted = ports.restart(old, fresh)
        for kept in ports.restart_keeps:
            np.testing.assert_array_equal(
                np.asarray(getattr(restarted, kept)),
                np.asarray(getattr(old, kept)),
            )
        # Everything else is the fresh state's, the centre included.
        np.testing.assert_array_equal(
            np.asarray(restarted.center), np.asarray(fresh.center)
        )

    def test_points_are_read_from_their_own_fields(self, problem):
        update_class, _ = _built(problem, "turbo")
        state = update_class.init_fn(
            jnp.zeros((update_class.popsize, DIM)), jr.key(0)
        )
        state = state.replace(center=jnp.full(DIM, 0.25))
        np.testing.assert_array_equal(
            np.asarray(update_class.switch_ports.read_point(state)),
            np.full(DIM, 0.25),
        )


def test_shape_is_a_valid_switch_output():
    # Every shape reader's output passes through one lax.switch, so the
    # layouts must agree leaf for leaf.
    states = {name: _es_state(name)[0] for name in ("cmaes", "snes")}
    readers = [_ports(name).read_shape for name in ("cmaes", "snes")]
    out = jax.lax.switch(
        1,
        [
            lambda _, r=r, s=states[n]: r(s)
            for r, n in zip(readers, ("cmaes", "snes"), strict=True)
        ],
        0,
    )
    assert out.diagonal.shape == (DIM,)
