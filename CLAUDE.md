# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**l2co-optimizers** holds the bare optimizers of the L2CO ecosystem:
- the name registry and the `UpdateClass` container;
- the `OptimizationStep` spec;
- the optax/evosax factories, the optimistix and scipy minimisers, IPOPT, plus SHADE, TuRBO, the RBF trust region, per-evaluation-key L-BFGS and random search;
- what each optimizer exposes to a switching loop: its state-transfer ports and `optimizer_parts` (the switching layer itself — `SubOpt`, the handshake policy, `menu_loss_and_grad` — is l2co's, l2co ADR 0019);
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
  - `ScipyUpdateClass` overrides `run` with one `jax.pure_callback` that hands the whole budget to `scipy.optimize.minimize`, runs realizations sequentially, and raises on `step` (ADR 0002). `ipopt` runs on the same driver, through casadi (ADR 0003). `ipopt` and `trustconstr` take `hessian="exact"`: the driver's `hessian(x)` serves `jax.hessian` of the loss, on the sample `x` was evaluated with, as one billed entry (ADR 0007).
  - **`RunState`** (`_src/core/run_state.py`) is here too, with `reset` / `batch_reset` / `run` / `batch_evaluate` and `evaluate` public (`batch_run` stays private in `_src`) (`_src/core/model_evaluation.py`): `RunState.init` takes an already-built `UpdateClass` (l2co's `init_run_state` resolves the `OptimizationStep` against a task), and the rest rewraps the `UpdateClass` loop. l2co re-exports them.
  - **What stays in l2co:** `RolloutWrapper`, `HistoryState`'s exports (`l2co.history_to_xarray` / `history_to_xarray_realizations` / `history_to_dataloader`, which need xarray and the ERT), and every meta-optimizer (`strategy_wrapper`, `meta_optimizer`, models).

## Commands

```bash
uv sync --extra tests --extra dev
make test            # uv run pytest (the real-Task battery lives in l2co)
make lint            # uv run ruff check
make docs            # uv run mkdocs build
```

The git hook calls `pre-commit`, which is not on PATH in the devcontainer. Run `uv run --with pre-commit pre-commit run`, then `git commit --no-verify`.

**Every PR runs CI** (`.github/workflows/pull_request.yml`): ruff (pinned to the pre-commit rev; bump the two together), pre-commit, `pytest -m "not slow"` on Linux/macOS x Python 3.12/3.13, the package build and the docs build, with `bessagroup/f3dasm` checked out next to the repo (the `../f3dasm` editable source) at `main` for a PR into `main` and `develop` otherwise. `build_docs.yml` builds the docs on pushes to `main`.

**Releases are automated** (`.github/workflows/release.yml`, `patrick-kidger/action_update_python_project`). On every push to `main`, if the `pyproject.toml` version is newer than PyPI's, it builds, runs the tests against the sdist and the wheel (f3dasm from PyPI, via `--no-sources`), uploads to PyPI, then pushes the `v<version>` tag and creates the GitHub release. To release, bump `version` in `pyproject.toml` and `CITATION.cff` on `develop`, then merge `develop` into `main`. Never tag or `gh release create` by hand: the action's tag push fails on an existing tag, *after* the PyPI upload.

## Architecture

- **`_src` layout:** `_src/core/` holds everything optimizer-agnostic — contract types (`typing`), `UpdateClass` and its run loop, `opt_history`, `history_state`, `batching`, `run_state`, `model_evaluation`, `loss`, `popsize`, `sampler`, `stopping_criteria`, `optimizer_schedule`, `experimentdata`, `utils`, `state_transfer`. `_src/` itself holds the optimizer implementations (`optax_implementations`, `evosax_implementations`, `optimistix_implementations`, `scipy_implementations`, `ipopt`, `lbfgs`, `shade`, `turbo`, `rbf_trust_region`, `random_search`) and the modules that must see all of them (`mapping` — the registry, and `optimizer_parts`). **`core` never imports from outside `core`**; `tests/test_core_layering.py` enforces it, including `TYPE_CHECKING` and function-local imports. A new module goes in `core/` only if it names no specific optimizer.
- **Public surface:** `src/l2co_optimizers/__init__.py`, one flat namespace mirroring `l2co_tasks`. The implementation lives in `_src/`. A name is public only if a sibling package (l2co, rl2co, l2co_experiments, l2co-tasks, crax-l2co — code, tests, notebooks or Hydra `_target_`s) uses it, or it is part of the bare-optimizer authoring surface (registry, `UpdateClass` and its contract types, loss helpers, state transfer, `optimizer_parts`). The per-library factories (`optax_update`, `evosax_*_update`, `lbfgs_update`, …), `SHADE`, the `l2co_native_*` / `CONSTRUCTOR_HYPERPARAMETERS` / `LINESEARCH_FEVAL_BOUND` tables, `resolve_popsize` and the transfer tables are private: the built-ins are reached by registry name, and this package's own tests import them from `_src`. Add a new public symbol to `__init__.py` and to `tests/test_optimizers_facade.py`'s origin table.
- **Registry** (`_src/mapping.py`): keys are stored under `normalize_key`, which strips non-alphanumerics and lowercases. Every factory follows one keyword convention: `model=`, `loss_fn=`, `pass_rng=`, `opt_hash=`, `bounded=`, `stop_fn=`, plus `**hyperparameters`. Meta-optimizers register in l2co, not here.
- **Factories:** they live next to their library.
  - `optax_update` / `optax_update_extra_kwargs` and the `optax_fn` closures are in `_src/optax_implementations.py`.
  - The evosax equivalents are in `_src/evosax_implementations.py`.
  - The built-ins each have their own module.
  - `UpdateClass` (`_src/core/update_class.py`) is the container plus the run loop; factories never subclass it to change how a run executes, except `RandomSearchUpdateClass` and `ScipyUpdateClass`.
- **`bounded=None` and `stop_fn=None`** mean unbounded and never-stop. Keep that true for any new closure factory.
- **Samplers** (`_src/core/sampler.py`, reached by name through `get_sampler`) are called as `sampler(key, params, n_samples)` by `reset` and, after a switch, by l2co's and rl2co's handshakes with the incumbent. All but one read `params` only for its shape. `relative_normal_sampling` (ADR 0004) reads the values: member 0 is exactly `params` and the rest are `x + std * max(|x|, 1) * N(0, 1)`, so a task that keeps a prescribed start as its model (CUTEst's `x0`) starts there. Called with the incumbent it duplicates it, which ADR 0004 accepts.
- **`OptimizationStep.name`** is a databank key.
  - Resolution order: `alias`, then a namer registered with `register_schedule_namer` (meta-optimizer packages register theirs), then the generic join.
  - `tests/test_conf_optimizers.py` locks every shipped config's names and hashes against `tests/data/schedule_names_l2co_f677d8c.json`.
  - Never regenerate that file to make a test pass.
- **Hydra configs:** `src/l2co_optimizers/conf/optimizers/*.yaml` are lists of `_target_: l2co_optimizers.OptimizationStep`, resolved through `pkg://l2co_optimizers.conf`. Meta-optimizer configs belong in `l2co_experiments`, not here.
- **The switching layer is l2co's** (l2co ADR 0019). This package owns only what an optimizer exposes to it: the state-transfer ports (`state_transfer`, built into every `UpdateClass` and every `*Parts`) and `optimizer_parts`. Never add a handshake decision, `SubOpt` or menu dispatch here; `tests/test_optimizers_facade.py::test_switching_layer_lives_in_l2co` fails if one comes back. `optimizer_parts` and the `UpdateClass` factories are two construction paths for one optimizer; `tests/test_optimizer_parts.py` keeps their popsize/family/`own_ask` in step. `PLAIN_RUN_OPTIMIZERS` (public) names the entries `optimizer_parts` refuses because their solver owns the loop (optimistix, scipy, IPOPT; ADRs 0001-0003, 0006): a caller building a menu from a configured suite, like l2co_experiments' uniform-random meta-policy, leaves them out.

## Conventions

- Ruff, line length 79, numpy docstrings. Update docstrings and `Attributes` lists in the same change as any behaviour change.
- Decisions taken before the split are l2co ADRs. Cite them as "l2co ADR 00NN" (`docs/agents/domain.md` lists them). New decisions specific to this package go in `docs/adr/`, numbered from 0001.
- Glossary: `CONTEXT.md`. Say *bare optimizer* vs *meta-optimizer*, and *menu*, not portfolio.
- Open TODOs: drop `flax` (used only for `flax.struct` in `shade` / `turbo` / `rbf_trust_region`); decide `normalize_key`'s home (`_src/core/utils.py`).

## Related repositories

Part of the L2CO ecosystem (Bessa Research Group): [l2co](https://github.com/bessagroup/L2CO), [rl2co](https://github.com/bessagroup/rl2co), [l2co-tasks](https://github.com/bessagroup/l2co-tasks), [l2co_experiments](https://github.com/bessagroup/l2co_experiments), [agentic-l2co](https://github.com/bessagroup/agentic-l2co), [bbob-jax](https://github.com/bessagroup/bbob-jax), [f3dasm](https://github.com/bessagroup/f3dasm).
