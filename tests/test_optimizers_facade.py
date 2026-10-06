"""The ``l2co_optimizers`` public surface (l2co ADR 0007 / 0016).

Locks the contract that everything needed to use or author an optimizer
is reachable from the package root and is the *same object* as its
``l2co_optimizers._src`` origin, and that the package stands alone:
importing it loads neither ``l2co`` nor ``l2co_tasks``.
"""

import importlib
import subprocess
import sys

import pytest

import l2co_optimizers as facade


def test_all_symbols_are_importable():
    missing = [name for name in facade.__all__ if not hasattr(facade, name)]
    assert not missing, f"facade advertises but does not expose: {missing}"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        # Registry & construction
        ("UpdateClass", "l2co_optimizers._src.update_class"),
        ("register_optimizer", "l2co_optimizers._src.mapping"),
        ("optimizer_mapping", "l2co_optimizers._src.mapping"),
        ("optimizers", "l2co_optimizers._src.mapping"),
        # Contract types
        ("InitFunction", "l2co_optimizers._src.typing"),
        ("StepFunction", "l2co_optimizers._src.typing"),
        ("StopFunction", "l2co_optimizers._src.typing"),
        ("InputParameters", "l2co_optimizers._src.typing"),
        ("OptState", "l2co_optimizers._src.typing"),
        ("SamplerFunction", "l2co_optimizers._src.typing"),
        ("PopSize", "l2co_optimizers._src.typing"),
        # Loss helpers
        ("OptHistory", "l2co_optimizers._src.opt_history"),
        ("vmapped_loss", "l2co_optimizers._src.loss"),
        ("vmapped_loss_and_grad", "l2co_optimizers._src.loss"),
        ("vmapped_loss_and_grad_with_rng", "l2co_optimizers._src.loss"),
        ("vmapped_loss_with_rng", "l2co_optimizers._src.loss"),
        # Sub-optimizer primitives
        ("SubOpt", "l2co_optimizers._src.sub_optimizer"),
        ("optstep_to_subopt", "l2co_optimizers._src.sub_optimizer"),
        ("resolve_popsize", "l2co_optimizers._src.sub_optimizer"),
        ("grad_sub_optimizer", "l2co_optimizers._src.sub_optimizer"),
        ("pop_sub_optimizer", "l2co_optimizers._src.sub_optimizer"),
        ("l2co_native_evosax", "l2co_optimizers._src.sub_optimizer"),
        ("CONSTRUCTOR_HYPERPARAMETERS", "l2co_optimizers._src.sub_optimizer"),
        # Handshake policy (docs/adr/0009)
        ("HandshakePolicy", "l2co_optimizers._src.handshake_policy"),
        ("HANDSHAKE_POLICY", "l2co_optimizers._src.handshake_policy"),
        (
            "DEFAULT_HANDSHAKE_POLICY",
            "l2co_optimizers._src.handshake_policy",
        ),
        ("handshake_policy_for", "l2co_optimizers._src.handshake_policy"),
        # Base optimizer factories / registries
        (
            "normalized_optax_normal",
            "l2co_optimizers._src.optax_implementations",
        ),
        (
            "normalized_optax_from_state",
            "l2co_optimizers._src.optax_implementations",
        ),
        ("random_search_update", "l2co_optimizers._src.random_search"),
        # Moved in from l2co with the package split
        ("TaskLike", "l2co_optimizers._src.typing"),
        ("OptimizationStep", "l2co_optimizers._src.optimizer_schedule"),
        (
            "register_schedule_namer",
            "l2co_optimizers._src.optimizer_schedule",
        ),
        ("RecentHistory", "l2co_optimizers._src.opt_history"),
        ("get_sampler", "l2co_optimizers._src.sampler"),
        ("random_sampling", "l2co_optimizers._src.sampler"),
        ("count_parameters", "l2co_optimizers._src.popsize"),
        (
            "create_schedules_experimentdata",
            "l2co_optimizers._src.experimentdata",
        ),
        # The run loop and the state it threads (l2co ADR 0017)
        ("BatchState", "l2co_optimizers._src.batching"),
        ("HistoryState", "l2co_optimizers._src.history_state"),
        ("Carry", "l2co_optimizers._src.typing"),
        ("RunResult", "l2co_optimizers._src.update_class"),
        # Read by l2co's bridge: random-search dispatch + plot categories
        ("RandomSearchUpdateClass", "l2co_optimizers._src.random_search"),
        ("random_search_mapping", "l2co_optimizers._src.random_search"),
        ("normalized_evosax", "l2co_optimizers._src.evosax_implementations"),
        ("evosax_mapping", "l2co_optimizers._src.evosax_implementations"),
        ("optax_mapping", "l2co_optimizers._src.optax_implementations"),
        ("lbfgs_mapping", "l2co_optimizers._src.lbfgs"),
        ("shade_mapping", "l2co_optimizers._src.shade"),
        ("turbo_mapping", "l2co_optimizers._src.turbo"),
        (
            "rbf_trust_region_mapping",
            "l2co_optimizers._src.rbf_trust_region",
        ),
    ],
)
def test_facade_symbol_is_its_src_origin(name, module):
    origin = getattr(importlib.import_module(module), name)
    assert getattr(facade, name) is origin


def test_package_imports_neither_l2co_nor_l2co_tasks():
    """The layering the split exists for (l2co ADR 0016).

    Run in a fresh interpreter: in this test session another module may
    already have imported ``l2co_tasks`` (the opt-in integration tests).
    """
    code = (
        "import sys, l2co_optimizers; "
        "bad = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'l2co', 'l2co_tasks'}); "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "", f"importing l2co_optimizers loaded: {out}"
