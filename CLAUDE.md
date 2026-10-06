# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**l2co-optimizers** holds the bare optimizers of the L2CO ecosystem:
- the name registry and the `UpdateClass` container;
- the `OptimizationStep` spec;
- the optax/evosax factories, plus SHADE, TuRBO, the RBF trust region, per-evaluation-key L-BFGS and random search;
- the strategy-facing layer (`SubOpt`, the handshake policy, state transfer, `menu_loss_and_grad`);
- the Hydra `optimizers` config group.

It was extracted from `l2co` (`bessagroup/L2CO@f677d8c`); see l2co ADR 0016. JAX/Equinox, Python 3.12+, `uv`-managed.

## Layering, enforced

```
l2co-tasks        l2co-optimizers        (neither imports the other)
        \            /
            l2co               <- the bridge: RolloutWrapper, exports, meta-optimizers
          /       \
     rl2co     agentic-l2co
```

- **Never import `l2co` or `l2co_tasks` from `src/`.** A ruff `TID251` banned-api rule fails the lint, and `tests/test_optimizers_facade.py` asserts it in a fresh interpreter.
- **There is no task here** (l2co ADR 0018). Factories take `model`, `loss_fn` and `pass_rng` as required keywords; `RunState.init` takes a built `UpdateClass` plus `model`, `dataset`, `batch_size`. Popsize callables take the dimensionality (`count_parameters(model)`). `tests/test_optimizers_facade.py` rejects any public callable with a `task` parameter.
- **The run loop lives here, as `UpdateClass` methods** (l2co ADR 0017): `init_state`, `step` / `batch_step`, `run`, `batch_run` (routes on `sequential_realizations`), `batch_run_fused`, `batch_run_sequential`. So do the states it threads: `BatchState`, the `HistoryState` buffer and `Carry`. The loop takes a plain `dict[str, Array]` dataset, never a task.
  - `RandomSearchUpdateClass` overrides `run` (chunked) and `batch_run` (always sequential, for peak memory).
  - **`RunState`** (`_src/run_state.py`) is here too, with `reset` / `batch_reset` / `run` / `batch_evaluate` and `evaluate` public (`batch_run` stays private in `_src`) (`_src/model_evaluation.py`): `RunState.init` takes an already-built `UpdateClass` (l2co's `init_run_state` resolves the `OptimizationStep` against a task), and the rest rewraps the `UpdateClass` loop. l2co re-exports them.
  - **What stays in l2co:** `RolloutWrapper`, `HistoryState`'s exports (`l2co.history_to_xarray` / `history_to_xarray_realizations` / `history_to_dataloader`, which need xarray and the ERT), and every meta-optimizer (`strategy_wrapper`, `meta_optimizer`, models).

## Commands

```bash
uv sync --extra tests --extra dev
make test            # uv run pytest (the real-Task battery lives in l2co)
make lint            # uv run ruff check
make docs            # uv run mkdocs build
```

The git hook calls `pre-commit`, which is not on PATH in the devcontainer. Run `uv run --with pre-commit pre-commit run`, then `git commit --no-verify`.

## Architecture

- **Public surface:** `src/l2co_optimizers/__init__.py`, one flat namespace mirroring `l2co_tasks`. The implementation lives in `_src/`. A name is public only if a sibling package (l2co, rl2co, l2co_experiments, l2co-tasks, crax-l2co — code, tests, notebooks or Hydra `_target_`s) uses it, or it is part of the bare-optimizer authoring surface (registry, `UpdateClass` and its contract types, loss helpers, state transfer). The per-library factories (`optax_update`, `evosax_*_update`, `lbfgs_update`, …), `SHADE`, the `l2co_native_*` tables and the handshake/transfer tables are private: the built-ins are reached by registry name, and this package's own tests import them from `_src`. Add a new public symbol to `__init__.py` and to `tests/test_optimizers_facade.py`'s origin table.
- **Registry** (`_src/mapping.py`): keys are stored under `normalize_key`, which strips non-alphanumerics and lowercases. Every factory follows one keyword convention: `model=`, `loss_fn=`, `pass_rng=`, `opt_hash=`, `bounded=`, `stop_fn=`, plus `**hyperparameters`. Meta-optimizers register in l2co, not here.
- **Factories:** they live next to their library.
  - `optax_update` / `optax_update_extra_kwargs` and the `optax_fn` closures are in `_src/optax_implementations.py`.
  - The evosax equivalents are in `_src/evosax_implementations.py`.
  - The built-ins each have their own module.
  - `UpdateClass` (`_src/update_class.py`) is the container plus the run loop; factories never subclass it to change how a run executes, except `RandomSearchUpdateClass`.
- **`bounded=None` and `stop_fn=None`** mean unbounded and never-stop. Keep that true for any new closure factory.
- **`OptimizationStep.name`** is a databank key.
  - Resolution order: `alias`, then a namer registered with `register_schedule_namer` (meta-optimizer packages register theirs), then the generic join.
  - `tests/test_conf_optimizers.py` locks every shipped config's names and hashes against `tests/data/schedule_names_l2co_f677d8c.json`.
  - Never regenerate that file to make a test pass.
- **Hydra configs:** `src/l2co_optimizers/conf/optimizers/*.yaml` are lists of `_target_: l2co_optimizers.OptimizationStep`, resolved through `pkg://l2co_optimizers.conf`. Meta-optimizer configs belong in `l2co_experiments`, not here.
- **Handshake and state transfer** (`handshake_policy`, `state_transfer`, `sub_optimizer`) are the single implementation shared by l2co's strategy and rl2co's env and wrappers. Do not fork a local copy downstream.

## Conventions

- Ruff, line length 79, numpy docstrings. Update docstrings and `Attributes` lists in the same change as any behaviour change.
- Decisions taken before the split are l2co ADRs. Cite them as "l2co ADR 00NN" (`docs/agents/domain.md` lists them). New decisions specific to this package go in `docs/adr/`, numbered from 0001.
- Glossary: `CONTEXT.md`. Say *bare optimizer* vs *meta-optimizer*, and *menu*, not portfolio.
- Open TODOs: drop `flax` (used only for `flax.struct` in `shade` / `turbo` / `rbf_trust_region`); decide `normalize_key`'s home (`_src/utils.py`).

## Related repositories

Part of the L2CO ecosystem (Bessa Research Group): [l2co](https://github.com/bessagroup/L2CO), [rl2co](https://github.com/bessagroup/rl2co), [l2co-tasks](https://github.com/bessagroup/l2co-tasks), [l2co_experiments](https://github.com/bessagroup/l2co_experiments), [agentic-l2co](https://github.com/bessagroup/agentic-l2co), [bbob-jax](https://github.com/bessagroup/bbob-jax), [f3dasm](https://github.com/bessagroup/f3dasm).
