"""The ``l2co_optimizers`` public surface (l2co ADR 0007 / 0016).

Locks the contract that everything needed to use or author an optimizer
is reachable from the package root and is the *same object* as its
``l2co_optimizers._src`` origin, and that the package stands alone:
importing it loads neither ``l2co`` nor ``l2co_tasks``.
"""

#                                                                       Modules
# =============================================================================

# Standard
import ast
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

# Third-party
import pytest

# Local
import l2co_optimizers as facade

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def test_all_symbols_are_importable():
    missing = [name for name in facade.__all__ if not hasattr(facade, name)]
    assert not missing, f"facade advertises but does not expose: {missing}"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        # Registry & construction
        ("UpdateClass", "l2co_optimizers._src.core.update_class"),
        ("register_optimizer", "l2co_optimizers._src.mapping"),
        ("optimizer_mapping", "l2co_optimizers._src.mapping"),
        ("optimizers", "l2co_optimizers._src.mapping"),
        ("normalize_key", "l2co_optimizers._src.core.utils"),
        # Contract types
        ("InitFunction", "l2co_optimizers._src.core.typing"),
        ("StepFunction", "l2co_optimizers._src.core.typing"),
        ("StopFunction", "l2co_optimizers._src.core.typing"),
        ("InputParameters", "l2co_optimizers._src.core.typing"),
        ("OptState", "l2co_optimizers._src.core.typing"),
        ("SamplerFunction", "l2co_optimizers._src.core.typing"),
        ("PopSize", "l2co_optimizers._src.core.typing"),
        # Loss helpers
        ("OptHistory", "l2co_optimizers._src.core.opt_history"),
        ("vmapped_loss", "l2co_optimizers._src.core.loss"),
        ("vmapped_loss_and_grad", "l2co_optimizers._src.core.loss"),
        ("vmapped_loss_with_rng", "l2co_optimizers._src.core.loss"),
        ("vmapped_loss_and_grad_with_rng", "l2co_optimizers._src.core.loss"),
        # What a switching loop drives (l2co ADR 0019)
        ("optimizer_parts", "l2co_optimizers._src.optimizer_parts"),
        ("PLAIN_RUN_OPTIMIZERS", "l2co_optimizers._src.optimizer_parts"),
        ("GradientParts", "l2co_optimizers._src.optimizer_parts"),
        ("PopulationParts", "l2co_optimizers._src.optimizer_parts"),
        ("step_fevals", "l2co_optimizers._src.optax_implementations"),
        # The family held only by plain-run entries (ADR 0002)
        (
            "FAMILY_DERIVATIVE_FREE",
            "l2co_optimizers._src.core.state_transfer",
        ),
        # Base optimizer factories / registries
        (
            "normalized_optax_normal",
            "l2co_optimizers._src.optax_implementations",
        ),
        # Moved in from l2co with the package split
        ("OptimizationStep", "l2co_optimizers._src.core.optimizer_schedule"),
        (
            "register_schedule_namer",
            "l2co_optimizers._src.core.optimizer_schedule",
        ),
        ("RecentHistory", "l2co_optimizers._src.core.opt_history"),
        ("get_sampler", "l2co_optimizers._src.core.sampler"),
        ("random_sampling", "l2co_optimizers._src.core.sampler"),
        (
            "relative_normal_sampling",
            "l2co_optimizers._src.core.sampler",
        ),
        ("count_parameters", "l2co_optimizers._src.core.popsize"),
        (
            "create_schedules_experimentdata",
            "l2co_optimizers._src.core.experimentdata",
        ),
        # The run loop and the state it threads (l2co ADR 0017)
        ("BatchState", "l2co_optimizers._src.core.batching"),
        ("HistoryState", "l2co_optimizers._src.core.history_state"),
        ("Carry", "l2co_optimizers._src.core.typing"),
        ("RunState", "l2co_optimizers._src.core.run_state"),
        ("reset", "l2co_optimizers._src.core.run_state"),
        ("batch_reset", "l2co_optimizers._src.core.run_state"),
        ("run", "l2co_optimizers._src.core.run_state"),
        ("batch_evaluate", "l2co_optimizers._src.core.run_state"),
        ("evaluate", "l2co_optimizers._src.core.model_evaluation"),
        ("RunResult", "l2co_optimizers._src.core.update_class"),
        # Per-library registries, read by l2co (plot categories)
        ("normalized_evosax", "l2co_optimizers._src.evosax_implementations"),
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
    already have imported ``l2co_tasks``.
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


def _public_callables():
    """Every public callable, plus the methods of every public class."""
    for name in facade.__all__:
        obj = getattr(facade, name)
        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                if callable(member) and not attr.startswith("__"):
                    yield f"{name}.{attr}", member
            yield name, obj
        elif callable(obj):
            yield name, obj
        elif isinstance(obj, dict):
            # The registries: every factory a name resolves to.
            for key, factory in obj.items():
                if callable(factory):
                    yield f"{name}[{key!r}]", factory


def test_no_public_callable_takes_a_task():
    """There is no task here: factories take ``model`` / ``loss_fn`` /
    ``pass_rng``, and turning a task into those is l2co's job.

    Ruff stops an ``l2co_tasks`` import; this stops the same coupling
    coming back as an untyped ``task`` parameter.
    """
    offenders = []
    for qualname, fn in _public_callables():
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if "task" in params:
            offenders.append(qualname)
    assert offenders == [], f"take a task: {offenders}"


# The switching layer moved to l2co (l2co ADR 0019). What an optimizer
# exposes to it -- state-transfer ports, ``optimizer_parts`` -- stays;
# deciding a handshake or dispatching a menu does not come back.
SWITCHING_LAYER = frozenset(
    {
        "SubOpt",
        "grad_sub_optimizer",
        "pop_sub_optimizer",
        "optstep_to_subopt",
        "menu_loss_and_grad",
        "HandshakePolicy",
        "HANDSHAKE_POLICY",
        "DEFAULT_HANDSHAKE_POLICY",
        "POPULATION_HANDSHAKES",
        "handshake_policy_for",
        "population_best",
        "population_best_one",
    }
)


def test_switching_layer_lives_in_l2co():
    src = Path(facade.__file__).parent
    defined = set()
    for path in src.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                defined.add((node.name, path.name))
            elif isinstance(node, ast.Assign):
                defined.update(
                    (target.id, path.name)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
    offenders = sorted(
        f"{name} ({file})" for name, file in defined if name in SWITCHING_LAYER
    )
    assert offenders == [], f"switching layer defined here: {offenders}"
    assert not SWITCHING_LAYER & set(facade.__all__)
