"""``register_schedule_namer``: the naming extension point (l2co ADR 0016).

``OptimizationStep.name`` resolves alias -> registered namer -> generic
``{optimizer}_{k=v}`` join. The package ships no namers; meta-optimizer
packages register theirs next to ``register_optimizer``.
"""

import pytest

from l2co_optimizers import OptimizationStep, register_schedule_namer
from l2co_optimizers._src import optimizer_schedule


@pytest.fixture
def namers():
    saved = dict(optimizer_schedule._SCHEDULE_NAMERS)
    yield optimizer_schedule._SCHEDULE_NAMERS
    optimizer_schedule._SCHEDULE_NAMERS.clear()
    optimizer_schedule._SCHEDULE_NAMERS.update(saved)


def test_ships_no_namers():
    assert optimizer_schedule._SCHEDULE_NAMERS == {}


def test_generic_join_without_a_namer():
    step = OptimizationStep(
        "toymeta", hyperparameters={"model": "/x/ckpt.eqx"}
    )
    assert step.name == "toymeta_model=/x/ckpt.eqx"


def test_registered_namer_replaces_the_generic_join(namers):
    register_schedule_namer("toymeta", lambda s: "toymeta_ckpt")
    step = OptimizationStep(
        "toymeta", hyperparameters={"model": "/x/ckpt.eqx"}
    )
    assert step.name == "toymeta_ckpt"


def test_alias_wins_over_a_registered_namer(namers):
    register_schedule_namer("toymeta", lambda s: "toymeta_ckpt")
    step = OptimizationStep("toymeta", hyperparameters={"alias": "run_42"})
    assert step.name == "run_42"


def test_namer_matches_the_optimizer_exactly(namers):
    register_schedule_namer("toymeta", lambda s: "named")
    assert OptimizationStep("toymeta2").name == "toymeta2"
