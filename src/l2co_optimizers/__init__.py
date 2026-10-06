"""
L2CO Optimizers - bare optimizers compatible with the L2CO library.

The single public surface for using and *authoring* optimizers: the name
registry and :class:`UpdateClass` every factory returns -- with the run
loop as its methods (``init_state``, ``run``, ``batch_run``) and the
state that loop threads (:class:`BatchState`, :class:`HistoryState`) --
the :class:`OptimizationStep` spec the Hydra configs instantiate, the
optimizer contract types, the per-library registries, and the
strategy-facing layer (``SubOpt``, handshake policy, state transfer, menu
evaluation) that meta-optimizers dispatch through, and :class:`RunState`
with its ``reset`` / ``run`` / ``batch_evaluate`` entry points, which
build a run from a resolved ``UpdateClass``. Meta-optimization
strategies themselves, and ``HistoryState``'s xarray/``DataLoader``
exports, live in ``l2co``
(l2co ADR 0017). The layout of this namespace is documented in
``docs/api.md``.

There is no task here. A factory takes ``model``, ``loss_fn`` and
``pass_rng`` as keywords, and :meth:`RunState.init` adds the ``dataset``
and ``batch_size``; turning an ``l2co_tasks.Task`` into those is
``l2co``'s job, so neither package imports the other.
"""

#                                                                       Modules
# =============================================================================

# Local
from l2co_optimizers._src.core.batching import BatchState
from l2co_optimizers._src.core.experimentdata import (
    create_schedules_experimentdata,
)
from l2co_optimizers._src.core.handshake_policy import (
    DEFAULT_HANDSHAKE_POLICY,
    HANDSHAKE_POLICY,
    HandshakePolicy,
    handshake_policy_for,
    population_best,
    population_best_one,
)
from l2co_optimizers._src.core.history_state import HistoryState
from l2co_optimizers._src.core.loss import (
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_with_rng,
)
from l2co_optimizers._src.core.model_evaluation import evaluate
from l2co_optimizers._src.core.opt_history import (
    OptHistory,
    RecentHistory,
)
from l2co_optimizers._src.core.optimizer_schedule import (
    ALIAS_HYPERPARAMETER,
    OptimizationStep,
    register_schedule_namer,
)
from l2co_optimizers._src.core.popsize import count_parameters, shade_popsize
from l2co_optimizers._src.core.run_state import (
    RunState,
    batch_evaluate,
    batch_reset,
    reset,
    run,
)
from l2co_optimizers._src.core.sampler import (
    constant_sampling,
    get_sampler,
    grid_sampling,
    normal_sampling,
    random_sampling,
    xavier_sampling,
)
from l2co_optimizers._src.core.state_transfer import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    CONF_EXACT,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    TransferBundle,
    build_transfer_fns,
)
from l2co_optimizers._src.core.typing import (
    AskFunction,
    Carry,
    InitFunction,
    InputParameters,
    LossFunction,
    OptHistoryType,
    OptState,
    PopSize,
    SamplerFunction,
    StepFunction,
    StopFunction,
    TransferReadFunction,
    TransferWriteFunction,
)
from l2co_optimizers._src.core.update_class import (
    RunResult,
    UpdateClass,
)
from l2co_optimizers._src.core.utils import normalize_key
from l2co_optimizers._src.evosax_implementations import normalized_evosax
from l2co_optimizers._src.lbfgs import lbfgs_mapping
from l2co_optimizers._src.mapping import (
    optimizer_mapping,
    optimizers,
    register_optimizer,
)
from l2co_optimizers._src.menu_eval import menu_loss_and_grad
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_normal,
    optax_mapping,
)
from l2co_optimizers._src.rbf_trust_region import rbf_trust_region_mapping
from l2co_optimizers._src.shade import shade_mapping, shade_update
from l2co_optimizers._src.sub_optimizer import (
    CONSTRUCTOR_HYPERPARAMETERS,
    SubOpt,
    grad_sub_optimizer,
    optstep_to_subopt,
    pop_sub_optimizer,
    resolve_popsize,
)
from l2co_optimizers._src.turbo import turbo_mapping

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "ALIAS_HYPERPARAMETER",
    "CONF_ABSENT",
    "CONF_ESTIMATED",
    "CONF_EXACT",
    "CONSTRUCTOR_HYPERPARAMETERS",
    "DEFAULT_HANDSHAKE_POLICY",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "HANDSHAKE_POLICY",
    "AskFunction",
    "BatchState",
    "Carry",
    "HandshakePolicy",
    "HistoryState",
    "InitFunction",
    "InputParameters",
    "LossFunction",
    "OptHistory",
    "OptHistoryType",
    "OptState",
    "OptimizationStep",
    "PopSize",
    "RecentHistory",
    "RunResult",
    "RunState",
    "SamplerFunction",
    "StepFunction",
    "StopFunction",
    "SubOpt",
    "TransferBundle",
    "TransferReadFunction",
    "TransferWriteFunction",
    "UpdateClass",
    "batch_evaluate",
    "batch_reset",
    "build_transfer_fns",
    "constant_sampling",
    "count_parameters",
    "create_schedules_experimentdata",
    "evaluate",
    "get_sampler",
    "grad_sub_optimizer",
    "grid_sampling",
    "handshake_policy_for",
    "lbfgs_mapping",
    "menu_loss_and_grad",
    "normal_sampling",
    "normalize_key",
    "normalized_evosax",
    "normalized_optax_normal",
    "optax_mapping",
    "optimizer_mapping",
    "optimizers",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "random_sampling",
    "rbf_trust_region_mapping",
    "register_optimizer",
    "register_schedule_namer",
    "reset",
    "resolve_popsize",
    "run",
    "shade_mapping",
    "shade_popsize",
    "shade_update",
    "turbo_mapping",
    "vmapped_loss",
    "vmapped_loss_and_grad",
    "vmapped_loss_with_rng",
    "xavier_sampling",
]
