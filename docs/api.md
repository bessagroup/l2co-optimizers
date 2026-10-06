# API Reference

The whole public surface is the flat `l2co_optimizers` namespace. It holds what the sibling packages use, plus what an author of a new bare optimizer needs (the registry, `UpdateClass` and its contract types, the loss helpers, state transfer); the per-library factories and tables behind the registry are private. It is organised here the way an optimizer author meets it: the registry and the container every factory returns, the loop that runs it, the spec that names an optimizer, the contract types, then the strategy-facing layer meta-optimizers dispatch through. The built-in optimizers are reached by registry name (`optimizer_mapping("shade")`), not by class.

## Registry & construction

::: l2co_optimizers.optimizer_mapping

::: l2co_optimizers.register_optimizer

::: l2co_optimizers.UpdateClass

::: l2co_optimizers.create_schedules_experimentdata

### Per-library registries

`optimizers` is the merge of one name -> factory dict per library. The ones l2co groups its plot categories by are exported on their own: `optax_mapping`, `normalized_evosax` (the normalized-name -> evosax class table), `normalized_optax_normal`, `lbfgs_mapping`, `shade_mapping`, `turbo_mapping` and `rbf_trust_region_mapping`.

## Running an optimizer

The run loop is a set of [`UpdateClass`](#l2co_optimizers.UpdateClass) methods (`init_state`, `step`, `run`, `batch_run`, `batch_run_fused`, `batch_run_sequential`; l2co ADR 0017). It takes a plain `dict[str, Array]` dataset and threads these states. Exporting a `HistoryState` to xarray or to a `DataLoader` is l2co's job (`l2co.history_to_xarray` and friends).

::: l2co_optimizers.BatchState

::: l2co_optimizers.HistoryState

### Run state

`RunState` bundles one optimizer's population, running best, optimizer state and `UpdateClass`. `RunState.init` builds it from an already-built `UpdateClass`, a `model`, a `dataset` and a `batch_size`; `reset` / `batch_reset` sample a fresh population, and `run` / `batch_evaluate` hand the state to the methods above. Resolving an `OptimizationStep` against a task is l2co's `init_run_state`.

::: l2co_optimizers.RunState

::: l2co_optimizers.reset

::: l2co_optimizers.run

::: l2co_optimizers.batch_evaluate

::: l2co_optimizers.evaluate

`Carry` is the scan carry, `(params, opt_state, key, batch_state, done, recent_history)`. `RunResult` is what every run driver returns, `(params, best_params, best_loss, opt_state, batch_state, history_state)`.

## Optimizer specification

::: l2co_optimizers.OptimizationStep

::: l2co_optimizers.register_schedule_namer

## Contract types

::: l2co_optimizers.OptHistory

::: l2co_optimizers.RecentHistory

## Loss evaluation

::: l2co_optimizers.vmapped_loss

::: l2co_optimizers.vmapped_loss_and_grad

::: l2co_optimizers.vmapped_loss_with_rng

::: l2co_optimizers.vmapped_loss_and_grad_with_rng

## Samplers

::: l2co_optimizers.get_sampler

::: l2co_optimizers.random_sampling

::: l2co_optimizers.normal_sampling

::: l2co_optimizers.xavier_sampling

::: l2co_optimizers.constant_sampling

::: l2co_optimizers.grid_sampling

## Population sizes

::: l2co_optimizers.count_parameters

::: l2co_optimizers.shade_popsize

## Built-in algorithms

::: l2co_optimizers.shade_update

## Optimizer parts

What a loop that switches between optimizers drives directly. The
switching layer itself (`SubOpt`, the handshake policy, menu
evaluation) lives in l2co (l2co ADR 0019).

::: l2co_optimizers.optimizer_parts

::: l2co_optimizers.GradientParts

::: l2co_optimizers.PopulationParts

::: l2co_optimizers.step_fevals

## State transfer

::: l2co_optimizers.TransferBundle

::: l2co_optimizers.build_transfer_fns
