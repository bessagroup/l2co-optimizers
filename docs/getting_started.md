# Getting started

Install from PyPI:

```bash
pip install l2co-optimizers
```

To develop it, clone it next to a checkout of [`f3dasm`](https://github.com/bessagroup/f3dasm). The repository is `uv`-managed, and its `[tool.uv.sources]` installs `f3dasm` from that sibling checkout, in editable mode:

```bash
git clone https://github.com/bessagroup/f3dasm.git
git clone https://github.com/bessagroup/l2co-optimizers.git
cd l2co-optimizers
uv sync
```

Build an optimizer from the registry and step it. A factory takes the problem as three keywords -- `model`, `loss_fn` and `pass_rng` -- never as a task object:

```python
import jax.numpy as jnp, jax.random as jr
from l2co_optimizers import optimizer_mapping

def sphere(x, **sample):
    return jnp.sum((x - 0.5) ** 2)

model = jnp.zeros(4)
cmaes = optimizer_mapping("cmaes")(
    model=model, loss_fn=sphere, pass_rng=False, opt_hash=1
)
params = jnp.repeat(model[None], cmaes.popsize, axis=0)
state = cmaes.init_fn(params, jr.key(0))
(params, state, key), history = cmaes.step_fn((params, state, jr.key(0)), sample={})
```

To run a full budget, wrap the built optimizer in a `RunState` and call `batch_evaluate`. It runs several independent realizations, each from its own sampled starting point, and needs no task:

```python
import equinox as eqx
from l2co_optimizers import BatchState, RunState, batch_evaluate, normal_sampling

adam = optimizer_mapping("adam")(
    model=model, loss_fn=sphere, pass_rng=False, opt_hash=2, learning_rate=0.05
)
run_state = RunState.init(
    adam, model=model, dataset={}, batch_size=None, key=jr.key(0)
)
run_state, batch_state, history = batch_evaluate(
    run_state=run_state,
    batch_state=BatchState.init(dataset={}, batch_size=None, key=jr.key(0)),
    static=eqx.filter(model, eqx.is_inexact_array, inverse=True),
    dataset={},
    loss_fn=sphere,
    sampler=normal_sampling,
    n_iterations=100,
    pass_rng=False,
    key=jr.split(jr.key(1), 5),  # five realizations
    verbose=False,
)
history.output_min.shape  # (5, 100): the lowest loss at each iteration
```

To run on an `l2co_tasks.Task`, use l2co: its `init_run_state` builds this `RunState` from an `OptimizationStep` and a task, and its `RolloutWrapper` wraps the whole loop. l2co is where a task meets an optimizer. To add your own optimizer, see [Register your own optimizer](register_optimizer.ipynb).

## Hydra optimizer configurations

The package ships ready-made `optimizers` config groups under `l2co_optimizers/conf/optimizers/`, installed as package data. Each YAML is a list of `OptimizationStep` specs:
- **single-optimizer sweeps:** `adam`, `sepcmaes`, `lr_sweep_pde`;
- **the optimistix minimisers at their defaults (plain runs only):** `optimistix`;
- **the scipy minimisers at their defaults (plain runs only):** `scipy`;
- **IPOPT at its defaults (plain runs only):** `ipopt`;
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

