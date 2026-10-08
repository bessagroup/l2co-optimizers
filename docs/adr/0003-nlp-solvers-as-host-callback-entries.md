# SLSQP, trust-constr and IPOPT run on the scipy host-callback driver

Three gradient-based nonlinear-programming (NLP) solvers join the registry
as plain-run entries:

- `slsqp` and `trustconstr`, in `_src/scipy_implementations.py`;
- `ipopt`, in `_src/ipopt.py`.

Like the ADR 0002 entries, none of them can be a sub-optimizer in a switching
menu or an rl2co action. SNOPT, the fourth solver usually named with these
three, is not added (see the end of this ADR).

**This reverses two rows of ADR 0002's "Not added" table.** ADR 0002 left
SLSQP and trust-constr out as duplicates. The overlap is real: no task has
constraints beyond a box, and every databank task leaves `bounded` empty. On
such a problem, each of the three is a quasi-Newton method we already have
in another form:

| Entry | Without a box, it is | Closest existing entry |
|---|---|---|
| `slsqp` | dense BFGS with an L1 merit line search | `bfgs` |
| `trustconstr` | a dense-BFGS trust region solved by projected conjugate gradients | `bfgs`, `trustkrylov` |
| `ipopt` | L-BFGS with IPOPT's filter line search and inertia correction | `lbfgs` |

They are added as named baselines. Engineering-design studies use these
solvers, so a databank comparison needs them under their own names. What
they add is their globalization (merit function, filter, trust region) and
their handling of a box. A new search behaviour is not the reason.

## Same driver, same rules

All three use ADR 0002's driver unchanged. That fixes the following rules:

- one evaluation (value and gradient together) is one history entry, billed 1;
- a run lasts its whole budget, re-evaluating the final point once the solver
  stops;
- failure ends the run at the best point;
- noisy and minibatched losses pass straight through;
- a `stop_fn` raises;
- the hyperparameters are the library's own scalar settings, at the
  library's defaults.

`ipopt` gets the driver by passing its own `Minimizer` to `_scipy_update`.
The `ScipyUpdateClass` name is therefore now slightly too narrow: it also
runs a non-scipy solver.

**Tolerances**, chosen as in ADR 0002. All three methods were run on
sphere, a 1e6-conditioned ellipsoid, Rosenbrock and Rastrigin, at d = 2, 10
and 40, with a 5,000-evaluation budget.

| Method | Setting | How it ends |
|---|---|---|
| SLSQP | `ftol=0` | "positive directional derivative for linesearch" on 8 of 12; runs the budget on Rastrigin and on 10-D Rosenbrock (at its local minimum) |
| trust-constr | `gtol=0, xtol=1e-15` | its trust radius falls below `xtol`, on all 12 |
| IPOPT | `tol=1e-300, acceptable_iter=0` | "solve succeeded" on 5 of 12 (sphere at every d, the ellipsoid and Rosenbrock at d = 2); runs the budget on the rest |

trust-constr's `xtol` cannot be 0. Its radius then reaches exactly zero,
every trial point equals the current point (which scipy has cached), and it
iterates forever without asking for an evaluation. IPOPT's `tol` must be
positive. `acceptable_iter=0` turns off IPOPT's "solved to acceptable level"
heuristic.

## A solver that stops evaluating is stopped

trust-constr loops without evaluating in two ways:

- with `xtol=0`;
- with a box that is not `keep_feasible`: on `[-1, 0.2]`, starting at the
  boundary, it reached `maxiter=500` after 21 evaluations.

A databank cell that hangs this way is killed at its wall-clock limit, and
the cell writes nothing. So `_Driver.stalled()` counts the solver's
iterations: trust-constr calls it from scipy's `callback`, IPOPT from its
iteration callback. After `_STALL_ITERATIONS = 1000` consecutive iterations
without a new evaluation, the run ends at the solver's current point, as if
the solver had stopped by itself. The ADR 0002 methods do not use the
guard: none of them was seen to loop.

## trust-constr

- **Hessian:** scipy's default when `hess` is not given, a dense `BFGS()`,
  passed explicitly with a fresh instance for each run. (`hessian="exact"`
  now also offers the exact Hessian: ADR 0007.)
- **Rejected:** finite-difference Hessian-vector products, as `trustkrylov`
  uses. They would turn the entry into the trust-ncg that ADR 0002
  rejected. SR1 is not exposed.
- **Box:** passed as `keep_feasible`. Without that, trust-constr evaluated
  outside the box (at 0.248 on `[-1, 0.2]`) and stalled, as above. With it,
  every iterate stays inside the box.
- **Hyperparameters:** `initial_tr_radius`, plus `initial_barrier_parameter`
  and `initial_barrier_tolerance`, which act only when there is a box.

## SLSQP

- **Box:** passed natively. SLSQP never evaluates outside it.
- **Hyperparameters:** none. Its only scalar options are a tolerance and a
  finite-difference step that a JAX gradient never uses.

## IPOPT, through casadi

**Why casadi.** cyipopt 1.7.0, the usual Python binding, is published as a
source distribution only. It needs a system IPOPT, and there is no Oscar
module for one; conda is not allowed on Oscar. casadi's manylinux wheels
(abi3, CPython 3.11+) bundle IPOPT 3.14.19 with MUMPS 5.8.2, and they
install with uv like any other package.

**casadi is a hard dependency** (`casadi>=3.7,<4`), as optimistix is
(ADR 0001) and scipy is (ADR 0002). It costs about what jax already costs:

| | casadi 3.8.1 | jaxlib 0.11.2 |
|---|---|---|
| Wheel (Linux x86_64) | 88.5 MB | 89.9 MB |
| Installed | 257 MB | 323 MB |
| Platforms | Linux x86_64 and aarch64, macOS arm64 and x86_64, Windows | the same |

casadi's only dependency is numpy. Its licence, LGPLv3+, does not affect a
BSD-3 package that imports it. An optional `[ipopt]` extra was rejected:

- the `ipopt` registry entry and config group would exist in every install,
  but building the entry would fail in an install without the extra. In the
  databank, that is a failed cell rather than an install error;
- every downstream environment, l2co_experiments first, would have to
  remember to request the extra;
- `test_ipopt.py` would skip silently wherever casadi was missing.

casadi is imported inside the IPOPT run, not at module level. Importing
`l2co_optimizers` therefore does not load it (about 0.13 s), and processes
that never run IPOPT never pay for it.

**Hessian.** IPOPT gets its own limited-memory quasi-Newton approximation.
IPOPT needs the Hessian as a matrix, and an exact one from JAX would cost
`d` gradients per step. (Offered after all, as `hessian="exact"`: ADR
0007.)

**One evaluation per new point.** IPOPT asks for the value and the gradient
separately, and it often asks again at the point it has just evaluated. Each
new point is evaluated once, value and gradient together, and the most recent
point is cached, as scipy's own `ScalarFunction` does. On the 3-D test
function, IPOPT made 221 value and 25 gradient requests, and these were
173 evaluations. A test reproduces a direct casadi IPOPT run's evaluation
sequence exactly.

**IPOPT cannot be aborted from inside an evaluation.** Once the budget is
spent, or the driver raises, every further request returns NaN. IPOPT
treats NaN as an evaluation error and backtracks. The iteration callback
then stops it, and the driver's exception is re-raised once IPOPT returns.
This costs about 1,000 discarded requests after the budget, a few
milliseconds in all. casadi's warning for each NaN is switched off
(`show_eval_warnings`).

**Box.** `bound_relax_factor=0`, so IPOPT never evaluates outside the box.

**Hyperparameters:**

- `limited_memory_max_history` (IPOPT's default 6);
- `mu_init` (default 0.1, which acts only when there is a box).

**IPOPT is weak at its defaults on ill-conditioned problems.** On the
1e6-conditioned ellipsoid it reaches 3e-5 at d = 10 and 0.76 at d = 40 after
5,000 evaluations. SLSQP and trust-constr reach 0 on both.

This is IPOPT's own behaviour, not the wrapper's. Running casadi's IPOPT
directly on the 10-D ellipsoid, with exact float64 gradients and no JAX or
driver involved:

- at the default history of 6, it reaches 4e-3 after 400 iterations,
  spending about 10 line-search evaluations per iteration;
- with a history of 20, it reaches 1e-27.

The entry keeps IPOPT's default, as every other entry keeps its library's.

## Cost

Measured on one core, single-threaded BLAS, with a cheap NumPy loss. The
figures are milliseconds per evaluation, including the method's own
overhead. As in ADR 0002, the allowance is about 14 ms per evaluation.

| d | 40 | 256 | 1024 | 3843 |
|---|---|---|---|---|
| SLSQP | 0.09 | 0.74 | 20.7 | 572 |
| trust-constr | 2.2 | 2.7 | 8.3 | 164 |
| IPOPT | 0.4 | 1.2 | 3.4 | 11 |

SLSQP's dense subproblem exceeds the allowance from about d = 1000, and
trust-constr's dense BFGS from somewhat above that. As with COBYQA, there is
no dimension cap. A cell above those sizes can exceed the wall-clock limit;
it then writes nothing and is re-submitted on every regeneration. IPOPT
stays inside the allowance up to d = 3843, though only just once the cost of
the JAX evaluation is added.

## Config groups

- The `scipy` group gains `slsqp` and `trustconstr`. Nothing downstream used
  the group yet.
- `ipopt` is its own group: it is not a scipy method.
- The golden schedule-name file gains the three new entries. No existing
  entry changes.

## SNOPT is not added

SNOPT is commercial: it needs a licence and, for every Python route, its
source or its library.

- casadi's wheel ships no SNOPT plugin (`casadi.has_nlpsol("snopt")` is
  `False`).
- pyOptSparse, the usual route, is not on PyPI, and it builds SNOPT from
  the licensed source.

With a licence, SNOPT would run on this same driver. Either casadi's
`snopt` plugin, built against `libsnopt7`, or pyOptSparse would take the
place of `_ipopt_minimize`.
