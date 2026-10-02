# Getting started

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

To run an optimizer on an `l2co_tasks.Task` over a full budget, with batching, realizations and the history reduction, use `l2co.RunState` / `l2co.RolloutWrapper`. To add your own optimizer, see [Register your own optimizer](register_optimizer.ipynb).

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

