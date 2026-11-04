"""Mapping from optimizer names to their UpdateClass implementations."""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Local
from l2co_optimizers._src.core.constraint_handling import (
    CONSTRAINT_HANDLING_LEVELS,
    CONSTRAINTS_BOX,
    CONSTRAINTS_GENERAL,
    CONSTRAINTS_NONE,
)
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.core.utils import normalize_key
from l2co_optimizers._src.evosax_implementations import evosax_mapping
from l2co_optimizers._src.ipopt import ipopt_mapping
from l2co_optimizers._src.lbfgs import lbfgs_mapping
from l2co_optimizers._src.optax_implementations import optax_mapping
from l2co_optimizers._src.optimistix_implementations import optimistix_mapping
from l2co_optimizers._src.random_search import random_search_mapping
from l2co_optimizers._src.rbf_trust_region import rbf_trust_region_mapping
from l2co_optimizers._src.scipy_implementations import scipy_mapping
from l2co_optimizers._src.scipy_stepped import scipy_stepped_mapping
from l2co_optimizers._src.shade import shade_mapping
from l2co_optimizers._src.turbo import turbo_mapping

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Registry of optimizer-name -> factory callables. Each value is a callable
# that, when called with the keyword contract used by ``RunState.init``
# (``**hyperparameters, task, opt_hash, bounded, stop_fn``), returns an
# ``UpdateClass`` instance. Built-in entries are the base optimizers from the
# optax/evosax/random-search/shade/lbfgs/optimistix/scipy/ipopt modules
# only.
# Meta-optimizers are NOT hardcoded here: l2co's own ``"l2co"`` strategy
# self-registers via ``register_optimizer`` from
# ``l2co._src.meta_optimizer`` (a side effect of
# ``import l2co``), exactly the way importing ``rl2co`` registers ``"rl2co"``.
# Keeping this back-edge out of the registry is what breaks the
# ``l2co_update -> strategy_wrapper -> optimizer registry`` import cycle — see
# ``l2co ADR 0007``.
#: Registry keys are stored under :func:`~l2co_optimizers._src.core.utils.
#: normalize_key`, which strips non-alphanumerics and lowercases. Both
#: lookup paths must agree on this: ``RunState.init`` resolves the raw
#: name a config wrote, while ``optimizer_parts.resolve_popsize`` resolves
#: an already-normalized one. Every single-word name (all 61 of the
#: originals) normalizes to itself, so this only starts to matter for a
#: name carrying an underscore -- where storing it raw makes the
#: optimizer usable directly but *not* as a meta-optimizer sub-step,
#: which fails at run time with "Optimizer 'x' not recognized" rather
#: than at registration.
optimizers: dict[str, Callable[..., UpdateClass]] = {
    normalize_key(name): factory
    for name, factory in (
        evosax_mapping
        | optax_mapping
        | random_search_mapping
        | rbf_trust_region_mapping
        | shade_mapping
        | turbo_mapping
        | lbfgs_mapping
        | optimistix_mapping
        | scipy_mapping
        | scipy_stepped_mapping
        | ipopt_mapping
    ).items()
}

#: The constraint-handling level of every built-in that is above
#: :data:`CONSTRAINTS_NONE` (ADR 0008); every other built-in clips. Keys
#: are normalized. :func:`register_optimizer` writes here too, so a
#: registered entry carries the level it declared.
_constraint_handling: dict[str, str] = {
    # The solver takes inequality, equality and bound constraints.
    "cobyla": CONSTRAINTS_GENERAL,
    "cobyqa": CONSTRAINTS_GENERAL,
    "ipopt": CONSTRAINTS_GENERAL,
    "slsqp": CONSTRAINTS_GENERAL,
    "trustconstr": CONSTRAINTS_GENERAL,
    # The solver takes the bounds itself.
    "lbfgsb": CONSTRAINTS_BOX,
    "powell": CONSTRAINTS_BOX,
    "tnc": CONSTRAINTS_BOX,
    # Candidates are repaired or clipped before the state learns from them.
    "rbftrustregion": CONSTRAINTS_BOX,
    "shade": CONSTRAINTS_BOX,
    "turbo": CONSTRAINTS_BOX,
    # No state: each draw is clipped and evaluated, nothing is carried over.
    "randomsearch": CONSTRAINTS_BOX,
}


def register_optimizer(
    name: str,
    factory: Callable[..., UpdateClass],
    *,
    constraint_handling: str = CONSTRAINTS_NONE,
) -> None:
    """Register an external optimizer factory under ``name``.

    This is the extension point that lets sibling packages (e.g. ``rl2co``)
    plug their own ``UpdateClass`` into l2co's optimizer registry *without*
    l2co taking a dependency on them. The registering package calls this at
    import time; ``RunState.init`` then resolves the name through
    ``optimizer_mapping`` like any built-in optimizer.

    Parameters
    ----------
    name : str
        Optimizer name to register under. Normalized (non-alphanumerics
        stripped, lowercased) to match the lookup performed by
        :func:`optimizer_mapping`, so a name with an underscore resolves
        the same way through every call path. Re-registering an existing
        name overwrites the previous factory.
    factory : Callable[..., UpdateClass]
        Factory callable that returns an ``UpdateClass``. It must accept
        the bare-factory keyword contract: ``model``, ``loss_fn``,
        ``pass_rng``, ``opt_hash``, ``bounded``, ``stop_fn`` and
        ``**hyperparameters``. Meta-optimizers, which need a full task,
        register with ``l2co.register_optimizer`` instead.
    constraint_handling : str, optional
        How far the optimizer can enforce constraints: one of
        :data:`CONSTRAINT_HANDLING_LEVELS` (ADR 0008). Defaults to
        :data:`CONSTRAINTS_NONE`, which is right for a factory that
        only clips to ``bounded``. Re-registering a name replaces its
        level along with its factory.

    Raises
    ------
    ValueError
        If ``constraint_handling`` is not one of the levels.
    """
    if constraint_handling not in CONSTRAINT_HANDLING_LEVELS:
        raise ValueError(
            f"constraint_handling must be one of "
            f"{CONSTRAINT_HANDLING_LEVELS}, got {constraint_handling!r}"
        )
    key = normalize_key(name)
    optimizers[key] = factory
    _constraint_handling[key] = constraint_handling


def optimizer_mapping(name: str) -> Callable[..., UpdateClass]:
    """
    Maps optimizer name to corresponding UpdateClass or function.

    Parameters
    ----------
    name : str
        Name of the optimizer, in any spelling that normalizes to a
        registered key -- ``"rbf_trust_region"`` and
        ``"rbftrustregion"`` are the same optimizer.

    Returns
    -------
    Callable[..., UpdateClass]
        Factory callable that constructs the corresponding ``UpdateClass``
        when called with the ``RunState.init`` keyword contract.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    try:
        return optimizers[normalize_key(name)]
    except KeyError:
        raise ValueError(f"Optimizer '{name}' not recognized.") from None


def constraint_handling(name: str) -> str:
    """
    How far the optimizer registered under ``name`` can enforce
    constraints (ADR 0008).

    Parameters
    ----------
    name : str
        Name of the optimizer, in any spelling that normalizes to a
        registered key.

    Returns
    -------
    str
        One of :data:`CONSTRAINT_HANDLING_LEVELS`:

        - :data:`CONSTRAINTS_NONE`: the optimizer only has its
          candidates clipped to ``bounded``, and its own state never
          learns that they were moved;
        - :data:`CONSTRAINTS_BOX`: it keeps its own state inside the
          box;
        - :data:`CONSTRAINTS_GENERAL`: its solver also takes inequality
          and equality constraints, though nothing passes it any yet
          (l2co ADR 0020).

        A name added to :data:`optimizers` without going through
        :func:`register_optimizer` is :data:`CONSTRAINTS_NONE`.

    Raises
    ------
    ValueError
        If optimizer name is not recognized.
    """
    key = normalize_key(name)
    if key not in optimizers:
        raise ValueError(f"Optimizer '{name}' not recognized.")
    return _constraint_handling.get(key, CONSTRAINTS_NONE)
