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

The tables group the optimizers by what they use from the objective and how they search. Each reference is the paper that introduced the method, or the paper its library's implementation follows.

### First-order gradient-based

These use the loss and its gradient.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `adabelief` | AdaBelief | optax | [Zhuang et al. (2020)](https://arxiv.org/abs/2010.07468) |
| `adadelta` | AdaDelta | optax | [Zeiler (2012)](https://arxiv.org/abs/1212.5701) |
| `adafactor` | Adafactor | optax | [Shazeer & Stern (2018)](https://arxiv.org/abs/1804.04235) |
| `adagrad` | AdaGrad | optax | [Duchi et al. (2011)](https://jmlr.org/papers/v12/duchi11a.html) |
| `adam` | Adam | optax | [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980) |
| `adamax` | AdaMax | optax | [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980) |
| `adamaxw` | AdaMax with decoupled weight decay | optax | [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) |
| `adamw` | AdamW | optax | [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) |
| `adan` | Adan | optax | [Xie et al. (2022)](https://arxiv.org/abs/2208.06677) |
| `amsgrad` | AMSGrad | optax | [Reddi et al. (2018)](https://openreview.net/forum?id=ryQu7f-RZ) |
| `fromage` | Fromage | optax | [Bernstein et al. (2020)](https://arxiv.org/abs/2002.03432) |
| `lamb` | LAMB | optax | [You et al. (2020)](https://arxiv.org/abs/1904.00962) |
| `lars` | LARS | optax | [You et al. (2017)](https://arxiv.org/abs/1708.03888) |
| `lion` | Lion | optax | [Chen et al. (2023)](https://arxiv.org/abs/2302.06675) |
| `nadam` | NAdam (Adam with Nesterov momentum) | optax | [Dozat (2016)](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) |
| `nadamw` | NAdamW (AdamW with Nesterov momentum) | optax | [Dozat (2016)](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ); [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) |
| `noisysgd` | Noisy SGD | optax | [Neelakantan et al. (2015)](https://arxiv.org/abs/1511.06807) |
| `nonlinearcg` | Nonlinear conjugate gradient (Polak-Ribiere by default; Fletcher-Reeves, Hestenes-Stiefel, Dai-Yuan) with a backtracking Armijo line search | optimistix | [Polak & Ribière (1969)](https://doi.org/10.1051/m2an/196903R100351) |
| `novograd` | NovoGrad | optax | [Ginsburg et al. (2019)](https://arxiv.org/abs/1905.11286) |
| `optimisticadam` | Optimistic Adam | optax | [Daskalakis et al. (2017)](https://arxiv.org/abs/1711.00141) |
| `optimisticgradientdescent` | Optimistic gradient descent | optax | [Mokhtari et al. (2019)](https://arxiv.org/abs/1901.08511) |
| `radam` | RAdam | optax | [Liu et al. (2020)](https://arxiv.org/abs/1908.03265) |
| `rmsprop` | RMSProp | optax | [Hinton (2012)](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |
| `rprop` | Rprop | optax | [Riedmiller & Braun (1993)](https://ieeexplore.ieee.org/document/298623) |
| `sgd` | SGD | optax | [Robbins & Monro (1951)](https://doi.org/10.1214/aoms/1177729586) |
| `signsgd` | signSGD | optax | [Bernstein et al. (2018)](https://arxiv.org/abs/1802.04434) |
| `sm3` | SM3 | optax | [Anil et al. (2019)](https://arxiv.org/abs/1901.11150) |
| `yogi` | Yogi | optax | [Zaheer et al. (2018)](https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf) |

### Newton methods

These also use second-order information: here, Hessian-vector products from finite differences of gradients.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `tnc` | Truncated Newton (TNC): a line-search Newton method on finite-difference Hessian-vector products | scipy | [Nash (1984)](https://doi.org/10.1137/0721052) |
| `trustkrylov` | Newton trust region with a Krylov (GLTR) subproblem solver; Hessian-vector products by finite differences of gradients | scipy | [Gould et al. (1999)](https://doi.org/10.1137/S1052623497322735) |

`ipopt` and `trustconstr`, listed under quasi-Newton methods, become Newton methods with `hessian="exact"`: they then use `jax.hessian` of the loss (see [ADR 0007](docs/adr/0007-exact-hessians-for-ipopt-and-trustconstr.md)).

### Quasi-Newton methods

These build an estimate of the Hessian from successive gradients.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `bfgs` | BFGS with a backtracking Armijo line search, or with `linesearch="wolfe"` scipy's strong-Wolfe (Moré–Thuente) search | optimistix + built-in | [Nocedal & Wright (2006)](https://doi.org/10.1007/978-0-387-40065-5); [Moré & Thuente (1994)](https://doi.org/10.1145/192115.192132) |
| `dfp` | DFP with a backtracking Armijo line search | optimistix | [Fletcher & Powell (1963)](https://doi.org/10.1093/comjnl/6.2.163) |
| `ipopt` | IPOPT: primal-dual interior point with a filter line search and a limited-memory quasi-Newton Hessian, or the exact Hessian with `hessian="exact"` | IPOPT, through casadi | [Wächter & Biegler (2006)](https://doi.org/10.1007/s10107-004-0559-y) |
| `lbfgs` | L-BFGS, with a fresh PRNG key per linesearch evaluation on stochastic objectives | optax + built-in | [Liu & Nocedal (1989)](https://doi.org/10.1007/BF01589116) |
| `lbfgsb` | L-BFGS-B: limited-memory BFGS with a Moré–Thuente line search and bounds handled inside the method (gradient projection) | scipy | [Byrd et al. (1995)](https://doi.org/10.1137/0916069) |
| `slsqp` | Sequential least-squares quadratic programming (SLSQP): SQP with a dense BFGS Hessian and an L1 merit line search | scipy | Kraft (1988), DFVLR-FB 88-28 |
| `trustconstr` | trust-constr: trust-region SQP (an interior-point method when there is a box) with a dense BFGS Hessian, or the exact Hessian with `hessian="exact"` | scipy | [Lalee et al. (1998)](https://doi.org/10.1137/S1052623493262993); [Byrd et al. (1999)](https://doi.org/10.1137/S1052623497325107) |

### Derivative-free, population-based

These evolve a set of points using loss values only.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `differentialevolution` | Differential Evolution | evosax | [Storn & Price (1997)](https://doi.org/10.1023/A:1008202821328) |
| `diffusionevolution` | Diffusion Evolution | evosax | [Zhang et al. (2024)](https://arxiv.org/abs/2410.02543) |
| `gesmrga` | GESMR-GA | evosax | [Kumar et al. (2022)](https://arxiv.org/abs/2204.04817) |
| `mr15ga` | MR15-GA | evosax | [Rechenberg (1978)](https://doi.org/10.1007/978-3-642-81283-5_8) |
| `neldermead` | Nelder-Mead downhill simplex; the population is the simplex (dimensionality + 1 vertices) | optimistix | [Nelder & Mead (1965)](https://doi.org/10.1093/comjnl/7.4.308) |
| `pso` | Particle Swarm Optimization | evosax | [Kennedy & Eberhart (1995)](https://doi.org/10.1109/ICNN.1995.488968) |
| `samrga` | SAMR-GA | evosax | [Clune et al. (2008)](https://doi.org/10.1371/journal.pcbi.1000187) |
| `shade` | SHADE, with optional turning-based mutation | built-in (evosax API) | [Tanabe & Fukunaga (2013)](https://doi.org/10.1109/CEC.2013.6557555); [Sun et al. (2020)](https://doi.org/10.3390/math8091565) |
| `simplega` | Simple GA | evosax | [Such et al. (2017)](https://arxiv.org/abs/1712.06567) |

### Derivative-free, distribution-based

These sample each generation from a search distribution, and update the distribution from the losses.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `ars` | Augmented Random Search | evosax | [Mania et al. (2018)](https://arxiv.org/abs/1803.07055) |
| `asebo` | ASEBO | evosax | [Choromanski et al. (2019)](https://arxiv.org/abs/1903.04268) |
| `cmaes` | CMA-ES | evosax | [Hansen & Ostermeier (2001)](https://doi.org/10.1162/106365601750190398) |
| `crfmnes` | CR-FM-NES | evosax | [Nomura & Ono (2022)](https://arxiv.org/abs/2201.11422) |
| `des` | Discovered ES | evosax | [Lange et al. (2023)](https://arxiv.org/abs/2211.11260) |
| `esmc` | ESMC | evosax | [Merchant et al. (2021)](https://arxiv.org/abs/2107.09661) |
| `gradientlessdescent` | Gradientless Descent | evosax | [Golovin et al. (2019)](https://arxiv.org/abs/1911.06317) |
| `guidedes` | Guided ES | evosax | [Maheswaranathan et al. (2018)](https://arxiv.org/abs/1806.10230) |
| `hillclimbing` | Hill climbing | evosax | — |
| `iamalgamfull` | iAMaLGaM (full covariance) | evosax | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) |
| `iamalgamunivariate` | iAMaLGaM (univariate) | evosax | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) |
| `lmmaes` | LM-MA-ES | evosax | [Loshchilov et al. (2017)](https://arxiv.org/abs/1705.06693) |
| `maes` | MA-ES | evosax | [Beyer & Sendhoff (2017)](https://doi.org/10.1109/TEVC.2017.2680320) |
| `noisereusees` | Noise-Reuse ES | evosax | [Li et al. (2023)](https://arxiv.org/abs/2304.12180) |
| `openes` | OpenAI-ES | evosax | [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864) |
| `persistentes` | Persistent ES | evosax | [Vicol et al. (2021)](https://arxiv.org/abs/2112.13835) |
| `pgpe` | PGPE | evosax | [Sehnke et al. (2008)](https://doi.org/10.1007/978-3-540-87536-9_40) |
| `rmes` | Rm-ES | evosax | [Li & Zhang (2018)](https://doi.org/10.1109/TEVC.2017.2765682) |
| `sepcmaes` | Sep-CMA-ES | evosax | [Ros & Hansen (2008)](https://hal.inria.fr/inria-00287367/document) |
| `simplees` | Simple ES | evosax | [Rechenberg (1978)](https://doi.org/10.1007/978-3-642-81283-5_8) |
| `simulatedannealing` | Simulated annealing | evosax | [Kirkpatrick et al. (1983)](https://doi.org/10.1126/science.220.4598.671) |
| `snes` | SNES | evosax | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) |
| `xnes` | xNES | evosax | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) |

### Surrogate-model-based

These fit a model to the points evaluated so far, and choose the next points from it.

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `cobyqa` | COBYQA: derivative-free trust region on quadratic interpolation models | scipy | [Ragonneau (2022)](https://theses.lib.polyu.edu.hk/handle/200/12294) |
| `rbf_trust_region` | RBF-surrogate trust-region search (ORBIT / DYCORS family) | built-in | [Wild & Shoemaker (2011)](https://doi.org/10.1137/09074927X); [Regis & Shoemaker (2013)](https://doi.org/10.1080/0305215X.2012.687731) |
| `turbo` | TuRBO trust-region Bayesian optimization | built-in | [Eriksson et al. (2019)](https://arxiv.org/abs/1910.01739) |

### Miscellaneous

| Name | Algorithm | Backend | Reference |
| --- | --- | --- | --- |
| `powell` | Powell's conjugate direction method (derivative-free line searches) | scipy | [Powell (1964)](https://doi.org/10.1093/comjnl/7.2.155) |
| `randomsearch` | One-shot random search | built-in | [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) |

The four optimistix entries (`bfgs`, `dfp`, `nonlinearcg`, `neldermead`) run as plain registry entries only: they evaluate the objective themselves, so `optimizer_parts` cannot unpack them into a switching menu. They bill the evaluations optimistix actually makes (one per step for the gradient solvers; for Nelder-Mead `n + 1` on the first step, 2 per step and `n + 3` on a shrink), never stop early on their own convergence test, and clip into `bounded` before each evaluation. See [ADR 0001](docs/adr/0001-optimistix-minimisers-as-plain-run-entries.md). `bfgs` with `linesearch="wolfe"` is scipy's BFGS made steppable: in float64 it evaluates the points scipy's BFGS evaluates, and it records the gradient at every point. See [ADR 0005](docs/adr/0005-strong-wolfe-line-search-for-bfgs.md).

The seven scipy entries (`cobyqa`, `powell`, `tnc`, `trustkrylov`, `slsqp`, `trustconstr`, `lbfgsb`) are plain registry entries too, for a different reason: scipy owns the optimization loop, so each run hands its whole budget to `scipy.optimize.minimize` inside one host callback. One iteration is one evaluation (value, or value and gradient), billed one. When scipy finishes before the budget, every remaining iteration re-evaluates its final point; if scipy fails, the run stays at the best point found. A `stop_fn` raises. See [ADR 0002](docs/adr/0002-scipy-minimisers-as-whole-run-callback-entries.md), and [ADR 0003](docs/adr/0003-nlp-solvers-as-host-callback-entries.md) for SLSQP and trust-constr, and [ADR 0006](docs/adr/0006-l-bfgs-b-as-a-named-baseline.md) for L-BFGS-B.

`ipopt` runs the same way, on the same driver, through casadi, whose wheels bundle IPOPT. IPOPT uses its own limited-memory quasi-Newton Hessian, and a value request and a gradient request at the same point are one evaluation. See [ADR 0003](docs/adr/0003-nlp-solvers-as-host-callback-entries.md). With `hessian="exact"`, `ipopt` and `trustconstr` use `jax.hessian` of the loss instead, taken on the sample its point was evaluated with; each Hessian is one billed evaluation. A loss computed outside JAX must supply its own Hessian (l2co-tasks ADR 0005). See [ADR 0007](docs/adr/0007-exact-hessians-for-ipopt-and-trustconstr.md).

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

Up to `l2co` 1.6.0 the sibling packages declared each other unpinned, so nothing enforced compatibility between releases. From 1.7.0, `l2co` declares `l2co-optimizers>=0.3.0`. The pairs:

| `l2co` | `l2co-optimizers` |
| --- | --- |
| 1.6.0 | 0.1.0 only |
| 1.7.0 | 0.3.0 or later |

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
