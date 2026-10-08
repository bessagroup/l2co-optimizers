# scipy minimisers run as one host callback per run

Four `scipy.optimize.minimize` methods join the registry under bare names:
`cobyqa`, `powell`, `tnc` and `trustkrylov`
(`_src/scipy_implementations.py`). They exist to add databank baselines that
cover three gaps in the registry:

- derivative-free local search: COBYQA (a trust region on quadratic
  interpolation models) and Powell (conjugate directions);
- truncated Newton: TNC;
- Newton trust region: trust-krylov.

They are plain-run entries, like the optimistix ones (ADR 0001). They can
never be a sub-optimizer in a switching menu or an rl2co action.

**scipy owns the loop, so the whole run is one callback.** Every one of these
methods calls the objective itself. TNC does it from C; the others from a
Python loop that keeps its state in local variables. None of them can be
stepped from a JAX scan. Only two scipy methods hand control back after each
evaluation: L-BFGS-B (`_lbfgsb.setulb`) and SLSQP (`_slsqplib.slsqp`). With
those two, the solver's workspace can ride in the scan carry. But L-BFGS-B
duplicates `lbfgs`, and on a box-only problem SLSQP is a dense BFGS, which
duplicates `bfgs`. So neither is added. `ScipyUpdateClass` overrides `run`
with a single `jax.pure_callback`. On the host it:

1. runs `minimize` against a jitted per-evaluation function;
2. returns one loss per iteration, the best point, the last point and the
   advanced batch state, all as fixed-length arrays;
3. lets `run` build the `HistoryState` with `from_reduced`, as random search
   does.

`batch_run` maps `run` over realizations one at a time. `step` raises, and
`init_fn` returns a scalar placeholder. Nothing in l2co or l2co_experiments
changes: a databank cell reaches these entries by name, through the jitted
`RolloutWrapper.batch_evaluate`.

Calling a jitted function from inside a callback works on CPU, including on
a single core, which is how a databank run step is sized. Databank runs are
CPU-only on both clusters (`JAX_PLATFORMS: cpu` on Oscar). GPU is untested.

**One iteration is one evaluation.** Every call scipy makes to the objective
is one history entry, billed 1, with population size 1: a value for COBYQA
and Powell, a value and gradient together for TNC and trust-krylov. A run
makes exactly `n_iterations` of them. The databank budget counts iterations
(`min(1000·d, 25000)`), so the budget is also the number of evaluations, as
for every optax entry. Counting one major scipy iteration instead, with its
evaluations billed (the `lbfgs` way, l2co ADR 0010), was rejected. Powell
spends about 10·d evaluations per iteration, which would run past the
wall-clock limit, and a fixed-shape history could not record every evaluated
point.

**trust-krylov's Hessian-vector products are finite differences of JAX
gradients.** `hessp(x, p) ≈ (g(x + εp) − g(x)) / ε`. The gradient at the
probe point is one more recorded, billed evaluation. The gradient at `x` is
the one scipy already made: the driver caches the last few non-probe
gradients. TNC computes its own products the same way, internally.

Exact products from JAX were rejected. Billing them to the next f/g record
would leave runtime unbounded: up to d products per outer step, about 96
million on the 3843-dimensional spirals task. Making each product its own
billed iteration would produce records with no loss. The step ε is sized
from the precision the loss is actually evaluated in (`√eps · max(1, |x|) /
|p|`), because a float64-sized step vanishes under float32 rounding. For
the same reason, TNC's `accuracy` option is set to that precision's machine
epsilon; under x64 this is TNC's default.

**A run lasts its whole budget.** Stopping tolerances are as tight as each
method tolerates, so scipy stops only when the method cannot progress. They
were chosen on sphere, a 1e6-conditioned ellipsoid, Rosenbrock and Rastrigin
at d = 2, 10 and 40:

| Method | Setting | How it ends |
|---|---|---|
| COBYQA | `final_tr_radius=1e-15` | on its trust-region radius |
| Powell | `xtol=1e-15, ftol=0` | when it finds no improvement |
| TNC | `ftol=xtol=gtol=0` | when its line search fails |
| trust-krylov | `gtol=0` | see below |

trust-krylov has no clean tolerance to find. Whatever its `gtol`, it stops
when its model stops predicting improvement, or it proposes a non-finite
step at a machine-precision optimum. The iteration and evaluation caps
scipy sees are the largest C `int`: TNC overflows on anything larger.

From the point scipy stops, every remaining iteration re-evaluates
scipy's final point (`res.x`). Each re-evaluation is real and billed 1, as
a converged `bfgs` keeps evaluating near its final point. Two rejected
alternatives:

- Stopping and padding the rest with NaN: ADR 0001 rules it out, because
  the ERT would score the solver as censored.
- Copying the last loss without evaluating, billed 0: an unsolved run would
  then be charged only what it spent before converging, the restart-friendly
  accounting COCO uses, applied to these entries and to nothing else.

Restarts were rejected: no other entry restarts, and they would make these
multi-start methods.

**Failure ends the run at the best point.** Three cases count as failure:

- scipy raises;
- scipy proposes a point that is not finite (it is never evaluated);
- scipy returns a final point that is not finite.

In each case the run spends its remaining budget re-evaluating the best point
evaluated so far. A NaN or infinite loss is passed to scipy unchanged, since
each method's handling of one is part of the method, and the running best
skips it. The stop reason is logged at debug level only: `HistoryState` has
no field for it.

**Noisy and minibatched losses pass straight through.** Each evaluation
draws its batch and key exactly as `UpdateClass.step` would:
`batch_state.next(key)`, then `new_key, eval_key = split(key)`. On a noisy
loss the methods' comparisons are noise-polluted, and the finite-difference
products in TNC and trust-krylov become noise divided by ε. That is accepted,
as in ADR 0001: on those tasks the databank measures how badly a method
copes with noise.

**Bounds.** COBYQA, Powell and TNC take the box natively, with the starting
point clipped into it first, so they never evaluate outside it.
trust-krylov takes no box, so its loss is evaluated at the projected point,
with the gradient taken through the projection, as in ADR 0001. A coordinate
that leaves the box therefore sees a zero gradient and zero curvature, and
the Krylov step never moves it again. No databank task has a box today: every
task config leaves `bounded` empty.

**No stopping criterion.** A `stop_fn` raises `ValueError` at construction:
nothing can check it between steps inside one callback. No databank run has
ever set one. Replaying the recorded losses through `stop_fn` after the run
would work for the loss- and gradient-based criteria, but not for the
covariance criterion, and nothing needs it.

**A fourth family.** COBYQA and Powell hold `FAMILY_DERIVATIVE_FREE`
(`"derivative_free"`): a unit population, and an iterate plus a
derivative-free local model, which is an interpolation set or a set of
search directions. TNC and trust-krylov hold `FAMILY_GRADIENT`. Only plain-run
entries hold the new family, so it has no transfer spec:
`transfer_spec_for` raises a `ValueError` for it, rather than a bare
`KeyError`. Like the optimistix entries, these four leave
`transfer_read_fn` and `transfer_write_fn` as `None`. `optimizer_parts`
refuses them by name. l2co does not re-export the new constant yet.

**Hyperparameters** are scipy's scalar algorithmic settings, defaulting to
scipy's values:

| Entry | Settings |
|---|---|
| `cobyqa` | `initial_tr_radius` |
| `tnc` | `eta`, `stepmx`, `max_cg_iterations` (scipy's `maxCGit`; `None` keeps scipy's default) |
| `trustkrylov` | `initial_trust_radius`, `max_trust_radius`, `eta` |
| `powell` | none |

Powell's only setting, `direc`, is an array, and an array cannot be hashed
into an `OptimizationStep`. Tolerances and every budget option are fixed and
hidden, so early stopping cannot come back through a config. The `scipy`
config group ships the four bare names.

**COBYQA's cost limits where it can run.** Measured on one core with a cheap
loss, including the method's own overhead:

| d | 10 | 40 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| COBYQA, ms per evaluation | 2.6 | 7.0 | 13.1 | ~270 per step after its 2d + 1-point warm-up | tens of seconds per step |

A databank run step allows 2.5 h for 25 realizations of up to 25,000
evaluations, about 14 ms per evaluation. TNC, trust-krylov and Powell stay
under 1.5 ms per evaluation up to d = 3843. COBYQA has no dimension cap and
ships in the same config group as the other three. An experiment that
includes the group runs COBYQA at every dimension of its task set, and from
about d = 64 upward its cells will hit the wall-clock limit. A cell that
times out writes nothing and is re-submitted on every regeneration.

**Dependency.** `scipy>=1.18,<2` is now declared; before, scipy came in
only through jax. These entries use only the public `minimize` API. No test
locks scipy's numerical behaviour across releases, but the tests do check
that the wrapper reproduces a direct `minimize` call at the installed
version. The l2co_experiments lockfile fixes which version a databank run
uses.

**Not added.**

| Method | Reason |
|---|---|
| L-BFGS-B, BFGS, CG, Nelder-Mead, differential evolution | Another implementation of `lbfgs`, `bfgs`, `nonlinearcg`, `neldermead` and `differentialevolution` (L-BFGS-B added after all, as a named baseline: ADR 0006) |
| SLSQP | Dense BFGS on a box-only problem (added after all, as a named baseline: ADR 0003) |
| Newton-CG | Truncated Newton, like TNC |
| trust-ncg | A Krylov trust region, like trust-krylov |
| trust-exact, dogleg | Need the full n×n Hessian |
| COBYLA | COBYQA's idea with a linear model |
| trust-constr | Duplicates trust-ncg without a box; slow Python (added after all, with a BFGS Hessian, as a named baseline: ADR 0003) |
| DIRECT, dual_annealing, shgo | Need a finite box, and no task set has one |
| basinhopping | A meta-optimizer around a duplicate local solver |
| brute | Exponential in d |
| `least_squares`, `root`, `linprog`, `minimize_scalar` | Need residuals, are linear, or are 1-D |
