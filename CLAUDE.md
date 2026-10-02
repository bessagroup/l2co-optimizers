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
            l2co               <- the bridge: RunState, run_, RolloutWrapper
          /       \
     rl2co     agentic-l2co
```

- **Never import `l2co` or `l2co_tasks` from `src/`.** A ruff `TID251` banned-api rule fails the lint, and `tests/test_optimizers_facade.py` asserts it in a fresh interpreter.
- **Tasks arrive only as `TaskLike`** (`model`, `loss_fn`, `pass_rng`; `_src/typing.py`). Derive the dimensionality with `count_parameters(task.model)`.
- **The execution layer stays in l2co:** `init_` / `step` / `run_` / `batch_run_*`, `RunState`, `BatchState`, `HistoryState`, `RolloutWrapper`. So does every meta-optimizer (`strategy_wrapper`, `meta_optimizer`, models). Do not add a run loop here; `tests/toy_tasks.run_steps` is a test-only stand-in.

## Commands

```bash
uv sync --extra tests --extra dev
make test            # uv run pytest (l2co_tasks integration module skips)
uv run --with-editable ../l2co-tasks pytest      # + real-Task integration
make lint            # uv run ruff check
make docs            # uv run mkdocs build
```

The git hook calls `pre-commit`, which is not on PATH in the devcontainer. Run `uv run --with pre-commit pre-commit run`, then `git commit --no-verify`.

## Architecture

- **Public surface:** `src/l2co_optimizers/__init__.py`, one flat namespace mirroring `l2co_tasks`. The implementation lives in `_src/`. Add new public symbols in both places, and add them to `tests/test_optimizers_facade.py`'s origin table.
- **Registry** (`_src/mapping.py`): keys are stored under `normalize_key`, which strips non-alphanumerics and lowercases. Every factory follows one keyword convention: `task=`, `opt_hash=`, `bounded=`, `stop_fn=`, plus `**hyperparameters`.
- **Factories:** they live next to their library.
  - `optax_update` / `optax_update_extra_kwargs` and the `optax_fn` closures are in `_src/optax_implementations.py`.
  - The evosax equivalents are in `_src/evosax_implementations.py`.
  - The built-ins each have their own module.
  - `UpdateClass` (`_src/update_class.py`) is a container only.
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
