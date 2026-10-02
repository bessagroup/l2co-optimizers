"""
L2CO Optimizers - bare optimizers compatible with the L2CO library.

The single public surface for using and *authoring* optimizers: the name
registry and :class:`UpdateClass` container every factory returns, the
:class:`OptimizationStep` spec the Hydra configs instantiate, the
optimizer contract types, the per-library registries, and the
strategy-facing layer (``SubOpt``, handshake policy, state transfer, menu
evaluation) that meta-optimizers dispatch through. Meta-optimization
strategies themselves, and the loop that runs an optimizer on a task,
live in ``l2co``. The layout of this namespace is documented in
``docs/api.md``.

Tasks reach this package only through :class:`TaskLike`, a structural
protocol that ``l2co_tasks.Task`` satisfies; neither package imports the
other.
"""

#                                                                       Modules
# =============================================================================

from l2co_optimizers._src.evosax_implementations import (
    evosax_distribution_update,
    evosax_mapping,
    evosax_population_update,
    normalized_evosax,
)
from l2co_optimizers._src.experimentdata import (
    create_schedules_experimentdata,
)
from l2co_optimizers._src.handshake_policy import (
    DEFAULT_HANDSHAKE_POLICY,
    HANDSHAKE_POLICY,
    OPT_STATE_VARIANTS,
    POPULATION_HANDSHAKES,
    POPULATION_VARIANTS,
    HandshakePolicy,
    handshake_policy_for,
    population_best,
    population_best_one,
    population_handshake_for,
)
from l2co_optimizers._src.lbfgs import (
    lbfgs_mapping,
    lbfgs_per_eval_key,
    lbfgs_update,
    scale_by_zoom_linesearch_per_eval_key,
)
from l2co_optimizers._src.loss import (
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
)
from l2co_optimizers._src.mapping import (
    optimizer_mapping,
    optimizers,
    register_optimizer,
)
from l2co_optimizers._src.menu_eval import (
    GRAD_EVALUATION_WIDTH,
    menu_loss_and_grad,
)
from l2co_optimizers._src.opt_history import (
    OptHistory,
    RecentHistory,
)
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
    optax_mapping,
    optax_update,
    optax_update_extra_kwargs,
)
from l2co_optimizers._src.optimizer_schedule import (
    ALIAS_HYPERPARAMETER,
    OptimizationStep,
    register_schedule_namer,
)
from l2co_optimizers._src.popsize import (
    count_parameters,
    shade_popsize,
    variable_popsize,
    variable_popsize_even,
)
from l2co_optimizers._src.random_search import (
    RandomSearchUpdateClass,
    random_search_mapping,
    random_search_update,
)
from l2co_optimizers._src.rbf_trust_region import rbf_trust_region_mapping
from l2co_optimizers._src.sampler import (
    constant_sampling,
    get_sampler,
    grid_sampling,
    normal_sampling,
    random_sampling,
    xavier_sampling,
)
from l2co_optimizers._src.shade import (
    SHADE,
    shade_mapping,
    shade_update,
)
from l2co_optimizers._src.state_transfer import (
    CONF_ABSENT,
    CONF_ESTIMATED,
    CONF_EXACT,
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    TRANSFER_OVERRIDES,
    TransferBundle,
    TransferSpec,
    build_transfer_fns,
    transfer_spec_for,
    write_none,
)
from l2co_optimizers._src.stopping_criteria import (
    STOPPING_CRITERIA,
)
from l2co_optimizers._src.sub_optimizer import (
    CONSTRUCTOR_HYPERPARAMETERS,
    LINESEARCH_FEVAL_BOUND,
    SubOpt,
    grad_sub_optimizer,
    l2co_native_evosax,
    l2co_native_optax,
    optstep_to_subopt,
    pop_sub_optimizer,
    resolve_popsize,
)
from l2co_optimizers._src.turbo import turbo_mapping
from l2co_optimizers._src.typing import (
    AskFunction,
    InitFunction,
    InputParameters,
    LossFunction,
    OptHistoryType,
    OptState,
    PopSize,
    SamplerFunction,
    StepFunction,
    StopFunction,
    TaskLike,
    TransferReadFunction,
    TransferWriteFunction,
)
from l2co_optimizers._src.update_class import (
    UpdateClass,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "ALIAS_HYPERPARAMETER",
    "AskFunction",
    "CONF_ABSENT",
    "CONF_ESTIMATED",
    "CONF_EXACT",
    "CONSTRUCTOR_HYPERPARAMETERS",
    "DEFAULT_HANDSHAKE_POLICY",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "GRAD_EVALUATION_WIDTH",
    "HANDSHAKE_POLICY",
    "HandshakePolicy",
    "InitFunction",
    "InputParameters",
    "LINESEARCH_FEVAL_BOUND",
    "LossFunction",
    "OPT_STATE_VARIANTS",
    "OptHistory",
    "OptHistoryType",
    "OptState",
    "OptimizationStep",
    "POPULATION_HANDSHAKES",
    "POPULATION_VARIANTS",
    "PopSize",
    "RandomSearchUpdateClass",
    "RecentHistory",
    "SHADE",
    "STOPPING_CRITERIA",
    "SamplerFunction",
    "StepFunction",
    "StopFunction",
    "SubOpt",
    "TRANSFER_OVERRIDES",
    "TaskLike",
    "TransferBundle",
    "TransferReadFunction",
    "TransferSpec",
    "TransferWriteFunction",
    "UpdateClass",
    "build_transfer_fns",
    "constant_sampling",
    "count_parameters",
    "create_schedules_experimentdata",
    "evosax_distribution_update",
    "evosax_mapping",
    "evosax_population_update",
    "get_sampler",
    "grad_sub_optimizer",
    "grid_sampling",
    "handshake_policy_for",
    "l2co_native_evosax",
    "l2co_native_optax",
    "lbfgs_mapping",
    "lbfgs_per_eval_key",
    "lbfgs_update",
    "menu_loss_and_grad",
    "normal_sampling",
    "normalized_evosax",
    "normalized_optax_from_state",
    "normalized_optax_normal",
    "optax_mapping",
    "optax_update",
    "optax_update_extra_kwargs",
    "optimizer_mapping",
    "optimizers",
    "optstep_to_subopt",
    "pop_sub_optimizer",
    "population_best",
    "population_best_one",
    "population_handshake_for",
    "random_sampling",
    "random_search_mapping",
    "random_search_update",
    "rbf_trust_region_mapping",
    "register_optimizer",
    "register_schedule_namer",
    "resolve_popsize",
    "scale_by_zoom_linesearch_per_eval_key",
    "shade_mapping",
    "shade_popsize",
    "shade_update",
    "transfer_spec_for",
    "turbo_mapping",
    "variable_popsize",
    "variable_popsize_even",
    "vmapped_loss",
    "vmapped_loss_and_grad",
    "vmapped_loss_and_grad_with_rng",
    "vmapped_loss_with_rng",
    "write_none",
    "xavier_sampling",
]
