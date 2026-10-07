# L2CO Optimizers

| [**GitHub**](https://github.com/bessagroup/l2co-optimizers)
| [**PyPI**](https://pypi.org/project/l2co-optimizers/)
| [**Documentation**](https://l2co-optimizers.readthedocs.io/)

Bare optimizers compatible with the L2CO library

***

## Summary

`l2co-optimizers` is the optimizer half of the [L2CO](https://github.com/bessagroup/l2co) ecosystem, as [`l2co-tasks`](https://github.com/bessagroup/l2co-tasks) is the task half. It provides:
- **a name registry** of ready-to-run optimizers: the optax gradient methods, the evosax distribution- and population-based algorithms, the optimistix and scipy minimisers, IPOPT, plus SHADE, TuRBO, an RBF trust region, per-evaluation-key L-BFGS and random search;
- **the `UpdateClass` container** every registry factory returns;
- **the `OptimizationStep` spec** that names an optimizer with its hyperparameters and stopping criteria;
- **ready-made Hydra optimizer configs**.

It also ships what each optimizer exposes to a loop that switches between optimizers: its state-transfer ports, and `optimizer_parts`, which unpacks it into an optax transform or an `(init, ask, tell)` triple. The switching layer itself (the `SubOpt` adapter, the handshake policy, menu-dispatched loss evaluation), the meta-optimization strategies (`l2co`, `rl2co`, `agentic-l2co`) and the bridge to tasks live in [`l2co`](https://github.com/bessagroup/l2co).

## Statement of need

Learning-to-optimize and optimizer-selection research needs many optimizers behind one calling convention, so that a selector can switch between them mid-run. `l2co-optimizers` provides that convention without the meta-learning stack:
- every optimizer is built by `optimizer_mapping(name)(model=..., loss_fn=..., pass_rng=..., opt_hash=..., bounded=..., stop_fn=...)`;
- every one runs a budget through the same `UpdateClass.run` and reports into the same `HistoryState`;
- every one except the scipy minimisers and IPOPT also steps through the same `(params, opt_state, key)` carry, reporting each step as an `OptHistory`. Those seven own their loop and run the whole budget in one call (ADRs 0002 and 0003).

It depends on neither `l2co` nor `l2co-tasks`, and has no notion of a task: factories take `model`, `loss_fn` and `pass_rng` as keywords. `l2co` is the bridge that unpacks an `l2co_tasks.Task` into them (l2co ADR 0018).

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
(params, state, key), history = cmaes.step_fn(
    (params, state, jr.key(0)), sample={}
)
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

To run on an `l2co_tasks.Task`, use l2co: its `init_run_state` builds this `RunState` from an `OptimizationStep` and a task, and its `RolloutWrapper` wraps the whole loop. l2co is where a task meets an optimizer. To add your own optimizer, see [Register your own optimizer](./docs/register_optimizer.ipynb).

## Available optimizers

Every optimizer below is built by name through `optimizer_mapping(name)`. Names are normalized (non-alphanumerics stripped, lowercased), so `"rbf_trust_region"` and `"rbftrustregion"` resolve to the same entry. Meta-optimizers (`l2co`, `rl2co`) are not built in: they register themselves when their package is imported.

| Name | Algorithm | Family | Backend |
| --- | --- | --- | --- |
| `adabelief` | AdaBelief | Gradient | optax |
| `adadelta` | AdaDelta | Gradient | optax |
| `adafactor` | Adafactor | Gradient | optax |
| `adagrad` | AdaGrad | Gradient | optax |
| `adam` | Adam | Gradient | optax |
| `adamax` | AdaMax | Gradient | optax |
| `adamaxw` | AdaMax with decoupled weight decay | Gradient | optax |
| `adamw` | AdamW | Gradient | optax |
| `adan` | Adan | Gradient | optax |
| `amsgrad` | AMSGrad | Gradient | optax |
| `fromage` | Fromage | Gradient | optax |
| `lamb` | LAMB | Gradient | optax |
| `lars` | LARS | Gradient | optax |
| `lion` | Lion | Gradient | optax |
| `nadam` | NAdam (Adam with Nesterov momentum) | Gradient | optax |
| `nadamw` | NAdamW (AdamW with Nesterov momentum) | Gradient | optax |
| `noisysgd` | Noisy SGD | Gradient | optax |
| `novograd` | NovoGrad | Gradient | optax |
| `optimisticadam` | Optimistic Adam | Gradient | optax |
| `optimisticgradientdescent` | Optimistic gradient descent | Gradient | optax |
| `radam` | RAdam | Gradient | optax |
| `rmsprop` | RMSProp | Gradient | optax |
| `rprop` | Rprop | Gradient | optax |
| `sgd` | SGD | Gradient | optax |
| `signsgd` | signSGD | Gradient | optax |
| `sm3` | SM3 | Gradient | optax |
| `yogi` | Yogi | Gradient | optax |
| `lbfgs` | L-BFGS, with a fresh PRNG key per linesearch evaluation on stochastic objectives | Quasi-Newton | optax + built-in |
| `bfgs` | BFGS with a backtracking Armijo line search | Quasi-Newton | optimistix |
| `dfp` | DFP with a backtracking Armijo line search | Quasi-Newton | optimistix |
| `nonlinearcg` | Nonlinear conjugate gradient (Polak-Ribiere by default; Fletcher-Reeves, Hestenes-Stiefel, Dai-Yuan) with a backtracking Armijo line search | Gradient | optimistix |
| `tnc` | Truncated Newton (TNC): a line-search Newton method on finite-difference Hessian-vector products | Newton-type | scipy |
| `trustkrylov` | Newton trust region with a Krylov (GLTR) subproblem solver; Hessian-vector products by finite differences of gradients | Newton-type | scipy |
| `slsqp` | Sequential least-squares quadratic programming (SLSQP): SQP with a dense BFGS Hessian and an L1 merit line search | Quasi-Newton | scipy |
| `trustconstr` | trust-constr: trust-region SQP (an interior-point method when there is a box) with a dense BFGS Hessian | Quasi-Newton | scipy |
| `ipopt` | IPOPT: primal-dual interior point with a filter line search and a limited-memory quasi-Newton Hessian (Wächter & Biegler 2006) | Quasi-Newton | IPOPT, through casadi |
| `ars` | Augmented Random Search | Distribution-based | evosax |
| `asebo` | ASEBO | Distribution-based | evosax |
| `cmaes` | CMA-ES | Distribution-based | evosax |
| `crfmnes` | CR-FM-NES | Distribution-based | evosax |
| `des` | Discovered ES | Distribution-based | evosax |
| `esmc` | ESMC | Distribution-based | evosax |
| `gradientlessdescent` | Gradientless Descent | Distribution-based | evosax |
| `guidedes` | Guided ES | Distribution-based | evosax |
| `hillclimbing` | Hill climbing | Distribution-based | evosax |
| `iamalgamfull` | iAMaLGaM (full covariance) | Distribution-based | evosax |
| `iamalgamunivariate` | iAMaLGaM (univariate) | Distribution-based | evosax |
| `lmmaes` | LM-MA-ES | Distribution-based | evosax |
| `maes` | MA-ES | Distribution-based | evosax |
| `noisereusees` | Noise-Reuse ES | Distribution-based | evosax |
| `openes` | OpenAI-ES | Distribution-based | evosax |
| `persistentes` | Persistent ES | Distribution-based | evosax |
| `pgpe` | PGPE | Distribution-based | evosax |
| `rmes` | Rm-ES | Distribution-based | evosax |
| `sepcmaes` | Sep-CMA-ES | Distribution-based | evosax |
| `simplees` | Simple ES | Distribution-based | evosax |
| `simulatedannealing` | Simulated annealing | Distribution-based | evosax |
| `snes` | SNES | Distribution-based | evosax |
| `xnes` | xNES | Distribution-based | evosax |
| `differentialevolution` | Differential Evolution | Population-based | evosax |
| `diffusionevolution` | Diffusion Evolution | Population-based | evosax |
| `gesmrga` | GESMR-GA | Population-based | evosax |
| `mr15ga` | MR15-GA | Population-based | evosax |
| `pso` | Particle Swarm Optimization | Population-based | evosax |
| `samrga` | SAMR-GA | Population-based | evosax |
| `simplega` | Simple GA | Population-based | evosax |
| `neldermead` | Nelder-Mead downhill simplex; the population is the simplex (dimensionality + 1 vertices) | Population-based | optimistix |
| `shade` | SHADE, with optional turning-based mutation (Tanabe & Fukunaga 2013; Sun et al. 2020) | Population-based | built-in (evosax API) |
| `turbo` | TuRBO trust-region Bayesian optimization (Eriksson et al. 2019) | Model-based | built-in |
| `rbf_trust_region` | RBF-surrogate trust-region search (ORBIT / DYCORS family) | Model-based | built-in |
| `cobyqa` | COBYQA: derivative-free trust region on quadratic interpolation models (Ragonneau & Zhang) | Model-based | scipy |
| `powell` | Powell's conjugate direction method (derivative-free line searches) | Direct search | scipy |
| `randomsearch` | One-shot random search | Random | built-in |

The four optimistix entries (`bfgs`, `dfp`, `nonlinearcg`, `neldermead`) run as plain registry entries only: they evaluate the objective themselves, so `optimizer_parts` cannot unpack them into a switching menu. They bill the evaluations optimistix actually makes (one per step for the gradient solvers; for Nelder-Mead `n + 1` on the first step, 2 per step and `n + 3` on a shrink), never stop early on their own convergence test, and clip into `bounded` before each evaluation. See [ADR 0001](docs/adr/0001-optimistix-minimisers-as-plain-run-entries.md).

The six scipy entries (`cobyqa`, `powell`, `tnc`, `trustkrylov`, `slsqp`, `trustconstr`) are plain registry entries too, for a different reason: scipy owns the optimization loop, so each run hands its whole budget to `scipy.optimize.minimize` inside one host callback. One iteration is one evaluation (value, or value and gradient), billed one. When scipy finishes before the budget, every remaining iteration re-evaluates its final point; if scipy fails, the run stays at the best point found. A `stop_fn` raises. See [ADR 0002](docs/adr/0002-scipy-minimisers-as-whole-run-callback-entries.md), and [ADR 0003](docs/adr/0003-nlp-solvers-as-host-callback-entries.md) for SLSQP and trust-constr.

`ipopt` runs the same way, on the same driver, through casadi, whose wheels bundle IPOPT. IPOPT uses its own limited-memory quasi-Newton Hessian, and a value request and a gradient request at the same point are one evaluation. See [ADR 0003](docs/adr/0003-nlp-solvers-as-host-callback-entries.md).

Three of these get expensive in high dimensions. COBYQA's cost per evaluation grows steeply with the dimensionality, and SLSQP's and trust-constr's do from about a thousand dimensions, so high-dimensional runs of these three can exceed a cluster's wall-clock limit.

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

## Releases

Sibling packages in this ecosystem declare each other unpinned, so nothing enforces compatibility between releases. Pair the versions by hand:

| `l2co` | `l2co-optimizers` |
| --- | --- |
| 1.6.0 | 0.1.0 only |
| `develop` (unreleased) | 0.2.0 or later |

**`l2co` 1.6.0 does not import with `l2co-optimizers` 0.2.0**, which an unpinned install now picks. 0.2.0 no longer exports 37 names that `l2co` 1.6.0 imports: the switching layer (`SubOpt`, the handshake policy) moved into `l2co` (l2co ADR 0019), and the per-library factories became private. With `l2co` 1.6.0, install `l2co-optimizers==0.1.0`.

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
- [l2co-optimizers](https://github.com/bessagroup/l2co-optimizers) — Bare optimizers (registry, `UpdateClass`, `OptimizationStep`, state transfer) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB (noiseless and noisy), CEC 2005 and CEC 2017 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
