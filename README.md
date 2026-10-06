![Bessa Research Group](img/bessa_group_logo.png)

# L2CO Optimizers

| [**GitHub**](https://github.com/bessagroup/l2co-optimizers)

Bare optimizers compatible with the L2CO library

***

## Summary

`l2co-optimizers` is the optimizer half of the [L2CO](https://github.com/bessagroup/l2co) ecosystem, as [`l2co-tasks`](https://github.com/bessagroup/l2co-tasks) is the task half. It provides:
- **a name registry** of ready-to-run optimizers: the optax gradient methods, the evosax distribution- and population-based algorithms, plus SHADE, TuRBO, an RBF trust region, per-evaluation-key L-BFGS and random search;
- **the `UpdateClass` container** every registry factory returns;
- **the `OptimizationStep` spec** that names an optimizer with its hyperparameters and stopping criteria;
- **ready-made Hydra optimizer configs**.

It also ships the *strategy-facing* layer meta-optimizers dispatch through: the `SubOpt` adapter, the per-optimizer handshake policy and state transfer, and menu-dispatched loss evaluation. The meta-optimization strategies themselves (`l2co`, `rl2co`, `agentic-l2co`) and the loop that runs an optimizer on a task stay in [`l2co`](https://github.com/bessagroup/l2co).

## Statement of need

Learning-to-optimize and optimizer-selection research needs many optimizers behind one calling convention, so that a selector can switch between them mid-run. `l2co-optimizers` provides that convention without the meta-learning stack:
- every optimizer is built by `optimizer_mapping(name)(task=..., opt_hash=..., bounded=..., stop_fn=...)`;
- every one steps through the same `(params, opt_state, key)` carry;
- every one reports into the same `OptHistory`.

It depends on neither `l2co` nor `l2co-tasks`. A task reaches it only through `TaskLike`, a structural protocol (`model`, `loss_fn`, `pass_rng`) that `l2co_tasks.Task` satisfies unchanged. `l2co` is the bridge that runs one on the other.

## Authorship

**Authors**:
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Authors affiliation:**
- Delft University of Technology (Bessa Research Group)

**Maintainer:**
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Maintainer affiliation:**
- Delft University of Technology (Bessa Research Group)

## Getting started

`l2co-optimizers` is `uv`-managed and depends on an editable install of a sibling [`f3dasm`](https://github.com/bessagroup/f3dasm) checkout, so lay the repositories out side by side before syncing:

```bash
git clone https://github.com/bessagroup/f3dasm.git
git clone https://github.com/bessagroup/l2co-optimizers.git
cd l2co-optimizers
uv sync
```

Build an optimizer from the registry and step it. Any object with `model`, `loss_fn` and `pass_rng` is a task:

```python
from dataclasses import dataclass
import jax.numpy as jnp, jax.random as jr
from l2co_optimizers import optimizer_mapping, TaskLike

@dataclass
class Sphere:
    model = jnp.zeros(4)
    pass_rng = False
    def loss_fn(self, x, **sample):
        return jnp.sum((x - 0.5) ** 2)

task = Sphere()
assert isinstance(task, TaskLike)

cmaes = optimizer_mapping("cmaes")(task=task, opt_hash=1)
params = jnp.repeat(task.model[None], cmaes.popsize, axis=0)
state = cmaes.init_fn(params, jr.key(0))
(params, state, key), history = cmaes.step_fn((params, state, jr.key(0)), sample={})
```

To run an optimizer on an `l2co_tasks.Task` over a full budget, with batching, realizations and the history reduction, use `RunState.init` and `batch_evaluate` (or l2co's `RolloutWrapper`, which bundles them with a task). To add your own optimizer, see [Register your own optimizer](./docs/register_optimizer.ipynb).

## Hydra optimizer configurations

The package ships ready-made `optimizers` config groups under `l2co_optimizers/conf/optimizers/`, installed as package data. Each YAML is a list of `OptimizationStep` specs:
- **single-optimizer sweeps:** `adam`, `sepcmaes`, `lr_sweep_pde`;
- **the portfolios used across the L2CO studies:** `small`, `medium`, `standard`, `standard_no_stopping`, `all`;
- **curated menus:** `headroom4`, `contrast`, `two_functions`, `gaussian_classification`, `pde`, `supercompressible`.

Add the package to a Hydra application's search path and select a group:

```yaml
hydra:
  searchpath:
    - pkg://l2co_optimizers.conf

defaults:
  - optimizers: medium   # any file in l2co_optimizers/conf/optimizers/
```

Hydra merges a group's options across search paths. So an application can keep its own `conf/optimizers/*.yaml` next to these, as `l2co_experiments` does for its meta-optimizer configs. `create_schedules_experimentdata` turns a composed group into an `f3dasm.ExperimentData` with one `OptimizationStep` per row.

## Releases

Sibling packages in this ecosystem declare each other unpinned, so nothing enforces compatibility between releases. **`l2co-optimizers` 0.1.0 must be released before, or together with, `l2co` 1.6.0**, which is the first `l2co` to depend on it.

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co-optimizers/issues) to report them.

## License

Copyright (c) 2026, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/l2co-optimizers/blob/main/LICENSE) for the full license text.

## Related repositories

This package is part of the L2CO ecosystem developed in the [Bessa Research Group](https://github.com/bessagroup). The repositories below work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co-optimizers](https://github.com/bessagroup/l2co-optimizers) — Bare optimizers (registry, `UpdateClass`, `OptimizationStep`, handshake and state transfer) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB (noiseless and noisy), CEC 2005 and CEC 2017 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
