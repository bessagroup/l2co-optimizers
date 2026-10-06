# API Reference

The whole public surface is the flat `l2co_optimizers` namespace. It is organised here the way an optimizer author meets it: the registry and the container every factory returns, the loop that runs it, the spec that names an optimizer, the contract types, then the built-in factories, and finally the strategy-facing layer meta-optimizers dispatch through.

## Registry & construction

::: l2co_optimizers.optimizer_mapping

::: l2co_optimizers.register_optimizer

::: l2co_optimizers.UpdateClass

::: l2co_optimizers.create_schedules_experimentdata

### Per-library registries

`optimizers` is the merge of one name -> factory dict per library, each also exported on its own: `optax_mapping`, `evosax_mapping` (built from `normalized_evosax`, the normalized-name -> evosax class table), `lbfgs_mapping`, `shade_mapping`, `turbo_mapping`, `rbf_trust_region_mapping` and `random_search_mapping`. Consumers use them to group optimizers by library, e.g. l2co's plot categories.

::: l2co_optimizers.RandomSearchUpdateClass

## Running an optimizer

The run loop is a set of [`UpdateClass`](#l2co_optimizers.UpdateClass) methods (`init_state`, `step`, `run`, `batch_run`, `batch_run_fused`, `batch_run_sequential`; l2co ADR 0017). It takes a plain `dict[str, Array]` dataset and threads these states. Exporting a `HistoryState` to xarray or to a `DataLoader` is l2co's job (`l2co.history_to_xarray` and friends).

::: l2co_optimizers.BatchState

::: l2co_optimizers.HistoryState

### Building a run from a task

`RunState` bundles one optimizer's population, running best, optimizer state and `UpdateClass`. `RunState.init` resolves an `OptimizationStep` through the registry against a `RunnableTaskLike`; `reset` samples a fresh population, and `run` / `batch_run` / `batch_evaluate` hand the state to the methods above. l2co's `RolloutWrapper` bundles them with a task.

::: l2co_optimizers.RunState

::: l2co_optimizers.reset

::: l2co_optimizers.run

::: l2co_optimizers.batch_run

::: l2co_optimizers.batch_evaluate

::: l2co_optimizers.evaluate

`Carry` is the scan carry, `(params, opt_state, key, batch_state, done, recent_history)`. `RunResult` is what every run driver returns, `(params, best_params, best_loss, opt_state, batch_state, history_state)`.

## Optimizer specification

::: l2co_optimizers.OptimizationStep

::: l2co_optimizers.register_schedule_namer

## Contract types

::: l2co_optimizers.TaskLike

::: l2co_optimizers.RunnableTaskLike

::: l2co_optimizers.OptHistory

::: l2co_optimizers.RecentHistory

## Loss evaluation

::: l2co_optimizers.vmapped_loss

::: l2co_optimizers.vmapped_loss_and_grad

::: l2co_optimizers.menu_loss_and_grad

## Samplers

::: l2co_optimizers.get_sampler

::: l2co_optimizers.random_sampling

::: l2co_optimizers.normal_sampling

::: l2co_optimizers.xavier_sampling

::: l2co_optimizers.constant_sampling

::: l2co_optimizers.grid_sampling

## Population sizes

::: l2co_optimizers.count_parameters

::: l2co_optimizers.variable_popsize

::: l2co_optimizers.variable_popsize_even

::: l2co_optimizers.shade_popsize

## Base optimizer factories

::: l2co_optimizers.optax_update

::: l2co_optimizers.optax_update_extra_kwargs

::: l2co_optimizers.evosax_distribution_update

::: l2co_optimizers.evosax_population_update

::: l2co_optimizers.random_search_update

## Built-in algorithms

::: l2co_optimizers.SHADE

::: l2co_optimizers.shade_update

::: l2co_optimizers.lbfgs_update

::: l2co_optimizers.lbfgs_per_eval_key

## Sub-optimizer adapter

::: l2co_optimizers.SubOpt

::: l2co_optimizers.optstep_to_subopt

::: l2co_optimizers.grad_sub_optimizer

::: l2co_optimizers.pop_sub_optimizer

::: l2co_optimizers.resolve_popsize

## Handshake policy

::: l2co_optimizers.HandshakePolicy

::: l2co_optimizers.handshake_policy_for

## State transfer

::: l2co_optimizers.TransferBundle

::: l2co_optimizers.TransferSpec

::: l2co_optimizers.build_transfer_fns

::: l2co_optimizers.transfer_spec_for
