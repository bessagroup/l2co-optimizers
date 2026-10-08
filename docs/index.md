# L2CO Optimizers

Bare optimizers compatible with the L2CO library.

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
| `trustconstr` | trust-constr: trust-region SQP (an interior-point method when there is a box) with a dense BFGS Hessian, or the exact Hessian with `hessian="exact"` | Quasi-Newton | scipy |
| `lbfgsb` | L-BFGS-B: limited-memory BFGS with a Moré–Thuente line search and bounds handled inside the method (gradient projection) | Quasi-Newton | scipy |
| `ipopt` | IPOPT: primal-dual interior point with a filter line search and a limited-memory quasi-Newton Hessian, or the exact Hessian with `hessian="exact"` (Wächter & Biegler 2006) | Quasi-Newton | IPOPT, through casadi |
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

The four optimistix entries (`bfgs`, `dfp`, `nonlinearcg`, `neldermead`) run as plain registry entries only: they evaluate the objective themselves, so `optimizer_parts` cannot unpack them into a switching menu. They bill the evaluations optimistix actually makes (one per step for the gradient solvers; for Nelder-Mead `n + 1` on the first step, 2 per step and `n + 3` on a shrink), never stop early on their own convergence test, and clip into `bounded` before each evaluation. See [ADR 0001](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0001-optimistix-minimisers-as-plain-run-entries.md).

The seven scipy entries (`cobyqa`, `powell`, `tnc`, `trustkrylov`, `slsqp`, `trustconstr`, `lbfgsb`) are plain registry entries too, for a different reason: scipy owns the optimization loop, so each run hands its whole budget to `scipy.optimize.minimize` inside one host callback. One iteration is one evaluation (value, or value and gradient), billed one. When scipy finishes before the budget, every remaining iteration re-evaluates its final point; if scipy fails, the run stays at the best point found. A `stop_fn` raises. See [ADR 0002](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0002-scipy-minimisers-as-whole-run-callback-entries.md), and [ADR 0003](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0003-nlp-solvers-as-host-callback-entries.md) for SLSQP and trust-constr, and [ADR 0006](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0006-l-bfgs-b-as-a-named-baseline.md) for L-BFGS-B.

`ipopt` runs the same way, on the same driver, through casadi, whose wheels bundle IPOPT. IPOPT uses its own limited-memory quasi-Newton Hessian, and a value request and a gradient request at the same point are one evaluation. See [ADR 0003](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0003-nlp-solvers-as-host-callback-entries.md). With `hessian="exact"`, `ipopt` and `trustconstr` use `jax.hessian` of the loss instead, taken on the sample its point was evaluated with; each Hessian is one billed evaluation. A loss computed outside JAX must supply its own Hessian (l2co-tasks ADR 0005). See [ADR 0007](https://github.com/bessagroup/l2co-optimizers/blob/develop/docs/adr/0007-exact-hessians-for-ipopt-and-trustconstr.md).

Three of these get expensive in high dimensions. COBYQA's cost per evaluation grows steeply with the dimensionality, and SLSQP's and trust-constr's do from about a thousand dimensions, so high-dimensional runs of these three can exceed a cluster's wall-clock limit.

To add your own optimizer to the registry, see [Register your own optimizer](register_optimizer.ipynb).
