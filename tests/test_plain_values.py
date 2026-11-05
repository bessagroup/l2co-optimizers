"""Plain config values for settings that are functions or objects (O10).

The dataset paper's protocol revision sets lengths as fractions of the box
(spec Section 3.4). Three such settings are not numbers on the optimizer:

- ``std_schedule`` is an evosax *constructor* argument (an optax schedule)
  on 12 algorithms; a plain number now becomes a constant schedule.
- ARS, ASEBO, NRES and PES take an inner ``optax.adam(1e-3)`` as their
  ``optimizer`` constructor argument; ``learning_rate`` now sets its rate.
- Adafactor's ``learning_rate`` is a function in the paper,
  ``min(1e-2, 1/sqrt(t))``; the string ``"relative_step"`` now names it.

The names stay plain (``openes_std_schedule=0.2``,
``adafactor_learning_rate=relative_step``), so a schedule's databank key is
the same in every process.
"""

from __future__ import annotations

import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from l2co_optimizers import BatchState, OptimizationStep, optimizer_mapping
from l2co_optimizers._src.optimizer_parts import optimizer_parts
from l2co_optimizers._src.plain_values import relative_step

from .toy_problems import sphere_problem

DIM = 4

STD_SCHEDULE_NAMES = [
    "ars",
    "asebo",
    "esmc",
    "guidedes",
    "hillclimbing",
    "iamalgamfull",
    "iamalgamunivariate",
    "noisereusees",
    "openes",
    "persistentes",
    "simplega",
    "simulatedannealing",
]
INNER_ADAM_NAMES = ["ars", "asebo", "noisereusees", "persistentes"]


def _build(name, **hyperparameters):
    return optimizer_mapping(name)(
        **hyperparameters,
        **sphere_problem(DIM),
        opt_hash=7,
        bounded=(None, None),
        stop_fn=None,
    )


def _init_state(update_class):
    population = jr.uniform(jr.key(0), (update_class.popsize, DIM))
    batch_state = BatchState.init(dataset={}, batch_size=None, key=jr.key(1))
    return update_class.init_state(
        params=population, batch_state=batch_state, dataset={}, key=jr.key(2)
    )


#                                                             std_schedule
# =============================================================================


@pytest.mark.parametrize("name", STD_SCHEDULE_NAMES)
def test_a_number_for_std_schedule_sets_the_constant_width(name):
    state = _init_state(_build(name, std_schedule=0.2))
    np.testing.assert_allclose(np.asarray(state.std), 0.2, rtol=1e-6)


@pytest.mark.parametrize("name", STD_SCHEDULE_NAMES)
def test_without_std_schedule_the_library_default_stays(name):
    state = _init_state(_build(name))
    np.testing.assert_allclose(np.asarray(state.std), 1.0, rtol=1e-6)


def test_std_schedule_is_refused_where_the_constructor_has_none():
    with pytest.raises(ValueError, match="std_schedule"):
        _build("cmaes", std_schedule=0.2)


#                                                           inner Adam's rate
# =============================================================================


def test_learning_rate_sets_the_inner_adam():
    import evosax.algorithms as ea

    from l2co_optimizers._src.plain_values import split_evosax_hyperparameters

    ctor, rest = split_evosax_hyperparameters(
        "ars", ea.ARS, {"learning_rate": 0.01, "std_schedule": 0.2}
    )
    assert rest == {}
    inner = ctor["optimizer"]
    grads = jnp.array([1.0, -2.0, 0.5])
    updates, _ = inner.update(grads, inner.init(jnp.zeros(3)))
    # Adam's first step is lr * sign(g) up to eps.
    np.testing.assert_allclose(updates, -0.01 * jnp.sign(grads), rtol=1e-5)


@pytest.mark.parametrize("name", INNER_ADAM_NAMES)
def test_inner_learning_rate_builds_and_steps(name):
    update_class = _build(name, learning_rate=0.01, std_schedule=0.2)
    state = _init_state(update_class)
    assert np.isfinite(np.asarray(state.std)).all()


@pytest.mark.parametrize("name", ["openes", "guidedes", "esmc", "cmaes"])
def test_learning_rate_is_refused_outside_the_inner_adam_methods(name):
    with pytest.raises(ValueError, match="learning_rate"):
        _build(name, learning_rate=0.01)


#                                                    Adafactor relative step
# =============================================================================


def test_relative_step_is_the_paper_schedule():
    """``min(1e-2, 1/sqrt(t))`` with t counted from 1."""
    assert float(relative_step(0)) == pytest.approx(1e-2)
    assert float(relative_step(9_999)) == pytest.approx(1e-2)
    assert float(relative_step(39_999)) == pytest.approx(1 / 200)
    assert float(relative_step(99_999)) == pytest.approx(1 / np.sqrt(1e5))


def test_adafactor_takes_relative_step_by_name():
    update_class = _build("adafactor", learning_rate="relative_step")
    state = _init_state(update_class)
    assert jax.tree.leaves(state)


def test_relative_step_is_refused_for_other_optax_optimizers():
    with pytest.raises(ValueError, match="relative_step"):
        _build("adam", learning_rate="relative_step")


def test_an_unknown_named_learning_rate_is_refused():
    with pytest.raises(ValueError, match="no_such_schedule"):
        _build("adafactor", learning_rate="no_such_schedule")


#                                                       the parts path agrees
# =============================================================================


@pytest.mark.parametrize(
    ("name", "hp"),
    [
        ("openes", {"std_schedule": 0.2}),
        ("ars", {"std_schedule": 0.2, "learning_rate": 0.01}),
        ("adafactor", {"learning_rate": "relative_step"}),
    ],
)
def test_optimizer_parts_accepts_the_same_plain_values(name, hp):
    step = OptimizationStep(optimizer=name, hyperparameters=hp)
    problem = sphere_problem(DIM)
    parts = optimizer_parts(
        step,
        model=problem["model"],
        loss_fn=problem["loss_fn"],
        pass_rng=problem["pass_rng"],
    )
    assert parts is not None


#                                                       names and hashes
# =============================================================================


def test_names_and_hashes_are_plain_and_stable_across_processes():
    steps = [
        ("openes", {"std_schedule": 0.2}),
        ("ars", {"std_schedule": 0.2, "learning_rate": 0.01}),
        ("adafactor", {"learning_rate": "relative_step"}),
    ]
    here = [
        OptimizationStep(optimizer=n, hyperparameters=hp) for n, hp in steps
    ]
    assert [s.name for s in here] == [
        "openes_std_schedule=0.2",
        "ars_std_schedule=0.2_learning_rate=0.01",
        "adafactor_learning_rate=relative_step",
    ]
    code = (
        "from l2co_optimizers import OptimizationStep as S;"
        "print([S(optimizer=n, hyperparameters=h).hash"
        f" for n, h in {steps!r}])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == str([s.hash for s in here])
