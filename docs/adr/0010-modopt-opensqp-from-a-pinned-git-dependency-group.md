---
status: proposed
---

# modOpt's OpenSQP runs on the scipy driver, from a pinned git dependency

OpenSQP is the sequential quadratic programming (SQP) solver developed
inside modOpt (Joshy & Hwang, LSDOlab). Each step solves a quadratic
model of the problem, built with a damped BFGS Hessian. It then searches
along the result on an augmented-Lagrangian merit function (one score
that trades the loss off against constraint violation): a Wolfe line
search first, and an l1-penalty backtracking search if that fails. It
joins the registry as `opensqp` (`_src/opensqp.py`, issue #37). Like
`ipopt`, it is a plain-run entry on ADR 0002's host-callback driver, so
it can never be a sub-optimizer in a switching menu or an rl2co action.

It is added for three reasons:

- **A second SQP solver that takes constraints.** Under ADR 0008's
  levels (PR #35) it is `general`, next to `slsqp`. Like every entry at
  that level, it gets only the box today: l2co refuses a task with
  inequalities or equalities (l2co ADR 0020).
- **It accepts steps differently from SLSQP.** SLSQP uses an L1 merit
  line search. A databank comparison of the two can separate "SQP does
  well" from "SLSQP's line search does well".
- **It is modOpt's own solver.** The `modopt` config group reproduces
  modOpt's CUTEst benchmark, which does not run OpenSQP.

## modOpt is installed from GitHub, outside the package's dependencies

modOpt cannot be a dependency of a package that is released to PyPI:

- it is on GitHub only (version 0.4.0);
- `modopt` on PyPI is an unrelated library (CEA's image-processing
  ModOpt), so `modopt>=…` would install the wrong package;
- PyPI rejects a direct URL dependency, and modOpt itself depends on
  `array_manager` from GitHub.

**Decision.** modOpt comes in through a uv dependency group, not a
dependency or an extra:

```toml
[dependency-groups]
opensqp = ["modopt", "highspy>=1.7", "qpsolvers>=4.3", "quadprog>=0.1.13"]

[tool.uv.sources]
modopt = {git = "https://github.com/LSDOlab/modopt.git", rev = "b9c9952…"}
```

- **The pin.** The source pins modOpt to commit `b9c9952` (0.4.0, 11
  October 2026). modOpt asks for `array_manager` at `main`, and
  `uv.lock` pins the commit that resolved, `ef9d2c4`. The QP solvers
  OpenSQP imports, `qpsolvers`, `quadprog` and `highspy`, have wheels
  for Linux and macOS on CPython 3.12 and 3.13.
- **What the wheel says.** A dependency group is not part of the
  package metadata. `uv build` writes no `Requires-Dist` for modOpt or
  the QP solvers, and `twine check` passes. The PyPI upload is
  unaffected.
- **Where it is installed.** `uv sync --group opensqp` installs it. PR
  CI does so, and sets `L2CO_OPTIMIZERS_REQUIRE_MODOPT=1`, so a missing
  modOpt fails `tests/test_opensqp.py` instead of skipping it. The
  release workflow tests the built wheel with `--no-sources` and without
  the group. That resolves (simulated locally), `test_opensqp.py` skips
  with the install command as its reason, and
  `tests/test_opensqp_dependency.py` still runs.
- **Without it, the factory raises.** `opensqp_update` imports modOpt
  and the QP solvers, and raises an `ImportError` that says how to
  install them. PyPI's unrelated `modopt` has no `OpenSQP`, so it fails
  the same way. Inside the run, the driver would take the error for a
  failed run and spend the budget at the starting point.
- **Licence.** modOpt is LGPL-3.0+. Like casadi (ADR 0003), importing it
  does not affect a BSD-3 package.

ADR 0003 rejected an optional extra for casadi: the entry would exist in
every install but fail to build in some, and its tests would skip
silently. Both still apply here, in a weaker form. The failure is loud
(an `ImportError` when the entry is built, not a failed run), and the
tests cannot skip in CI. What remains is that every environment that runs
`opensqp`, l2co_experiments first, must install modOpt itself.

### Rejected

- *Reimplementing OpenSQP* in `_src/`, from its description. Its
  module is 1,311 lines, 636 of them the main loop: penalty updates,
  slack resets, an elastic mode, Hessian resets. A reimplementation
  that matched modOpt's iterates would be a port of LGPL code into a
  BSD-3 package. One that did not would not be OpenSQP.
- *Copying modOpt's source in.* The same licence problem.
- *Waiting for modOpt on PyPI.* Its `pyproject.toml` carries a
  commented-out link for `modopt-lib`, a name that is free on PyPI
  today. If that is published, the group switches to it and nothing else
  changes.

## Same driver, same rules

`opensqp` uses ADR 0002's driver unchanged, with the same rules as
`ipopt`:

- a run lasts its whole budget, re-evaluating OpenSQP's final point once
  it stops;
- failure ends the run at the best point;
- noisy and minibatched losses pass straight through;
- a `stop_fn` raises;
- the family is `FAMILY_GRADIENT`.

modOpt's `ProblemLite` wraps the driver's objective. The entry
subclasses `OpenSQP` only to hook the stall guard below into it, and
sets no option beyond those below. modOpt's output files are turned off
(`turn_off_outputs=True`), so a run writes nothing.

**Billing.** OpenSQP asks for the value and the gradient separately, at
the same point. Each new point is evaluated once, value and gradient
together, and billed 1, as for IPOPT. Unlike `ipopt`, which keeps only
the latest point, the entry keeps the last 64 (`_RECENT_POINTS`).

At a box optimum OpenSQP can repeat the same few points forever, a few
units in the last place apart. With only eight points kept, a boxed
256-dimensional Rosenbrock reached the box optimum within 100
evaluations. It then spent the next 3,075, until a 60-second limit,
trying the same 9 points on every iteration. Each was a new billed
evaluation, at 19 ms each, since every iteration also solves a dense
quadratic program. That is above the roughly 14 ms a databank run step
allows, and it would have gone on to the end of the budget. With 64
kept, the same run evaluated 968 points, then stopped asking for new
ones, and the stall guard below ended it.

One iteration evaluates at most about 22 points: 10 in the Wolfe line
search, 11 in the l1-penalty fallback, and the new iterate. 64 covers a
cycle of two iterations. On a noisy loss, a point asked for again within
those 64 returns the value from its first evaluation. The driver already
does this for the gradients behind trust-krylov's Hessian-vector
products.

**It never evaluates outside the box.** The bounds enter the quadratic
sub-problem as inequality rows, the step is clipped to the box, and the
line search never steps past the clipped step. `tests/test_opensqp.py`
checks that every evaluated point lies in the box, from a start outside
it.

## Stopping: tolerances of zero, and the stall guard

`opt_tol = feas_tol = 0`, and `maxiter` is the largest C `int`, chosen
as in ADRs 0002 and 0003, so OpenSQP reports convergence only at an
exactly stationary point. On sphere, a 1e6-conditioned ellipsoid,
Rosenbrock and Rastrigin, at d = 2, 10 and 40, through the entry in
float64, with a 5,000-evaluation budget:

| How it ends | Runs |
|---|---|
| Converges: the gradient is exactly zero | the ellipsoid at d = 2, Rosenbrock at every d |
| Proposes a non-finite point at f = 0 | sphere at every d, the ellipsoid at d = 10 and 40 |
| The stall guard | Rastrigin at d = 2 (after 306 evaluations); all twelve with a box |
| Runs the budget | Rastrigin at d = 10 and 40 |

At f = 0 the merit-function line search divides by zero and proposes a
non-finite point. The driver never evaluates it: the run ends at its
best point, f = 0, as `lbfgsb` does on the 40-D sphere (ADR 0006).

**The stall guard.** With a box, OpenSQP reaches the box optimum in 3 to
840 evaluations, then keeps iterating without asking for a new point.
The driver's stall guard (ADR 0003) ends the run after 1,000 such
iterations. The hook is OpenSQP's `update_outputs`, which it calls once
per iteration. Every one of the twelve boxed runs ended this way, at
the loss `lbfgsb` reaches from the same start, to six digits. OpenSQP
can resume after an idle stretch: once, on the 40-D sphere, it
evaluated again after 136 idle iterations. So the shared threshold of
1,000 stays.

## No wall-clock limit

modOpt solves each quadratic sub-problem with quadprog. If quadprog
fails, OpenSQP falls back to a relaxed sub-problem solved by HiGHS under
a wall-clock limit, `aqp_time_limit`, 5 s by default. That happens on a
box: at a box optimum OpenSQP's damped BFGS matrix reaches a condition
number near 3e15, and quadprog reports "constraints are inconsistent"
for a sub-problem that `p = 0` satisfies. In the twelve boxed runs it did
so 5,003 times.

A limit that can bind makes a run depend on how busy the machine is. The
entry sets `aqp_time_limit = inf`, so HiGHS always finishes. The
databank cell's own wall-clock limit is the backstop.

## Hyperparameters

None are exposed: every setting stays at modOpt's default, including
the line search's (`ls_*`). `hvp_mode`, OpenSQP's Hessian updates from
Hessian-vector products, stays off. `"fd"` takes finite differences at a
fixed step of 1e-6, which vanishes under float32 rounding (ADR 0002
sizes its steps for that). `"exact"` would need a Hessian-vector product
served by the driver.

The tolerances, `maxiter` and `aqp_time_limit` are fixed and hidden, as
for every plain-run entry.

## Cost

Measured on one core with single-threaded BLAS and a cheap NumPy loss
(Rosenbrock), with the entry's settings. Without a box, a run costs
milliseconds per evaluation. With a box, it costs that until the stall,
then 1,000 idle iterations, each solving a quadratic program, before the
guard ends the run:

| d | 40 | 256 | 1024 | 3843 |
|---|---|---|---|---|
| No box, ms per evaluation | 3.7 | 3.9 | 131 | 4,400 |
| Box, ms per evaluation until the stall | 1.8 | 40 | 2,850 | 106,000 |
| Box, ms per idle iteration | 2.3 | 108 | not reached | not reached |

The 3843-dimensional figures cover the first 411 evaluations without a
box and the first 20 with one, within a 30-minute limit. At d = 1024
with a box the run had not stalled after 632 evaluations and 30 minutes.

A databank run step allows about 14 ms per evaluation (ADR 0002).
Without a box, `opensqp` fits up to a few hundred dimensions and runs
out of time from about d = 1024. With a box, the quadratic programs
carry two bound rows per variable, and quadprog's failures send
OpenSQP to HiGHS on most iterations: already at d = 256 it costs 40 ms
an evaluation, and the 1,000 idle iterations about 108 s per
realization. There is no dimension cap, as for COBYQA, SLSQP and
trust-constr (ADRs 0002, 0003): a cell above those sizes will hit the
wall-clock limit.

## Side effects of importing modOpt

- **Warnings.** Importing modOpt prints invalid-escape, scipy-deprecation
  and "matplotlib not found" warnings from modules OpenSQP does not use.
  The factory silences them for the import only. During a run it
  silences two that repeat on every idle step: qpsolvers converting the
  relaxed sub-problem, and scipy's BFGS skipping a zero update.
- **matplotlib.** modOpt calls `matplotlib.use("TkAgg")` at import. On a
  machine without a display, a process that imports modOpt before
  pyplot can no longer import pyplot. Here that cannot happen: f3dasm,
  which `l2co_optimizers` imports, loads pyplot first, so modOpt's call
  fails quietly and the backend stays as it was.

## Config group

A new `opensqp` group holds the one default entry, as the `ipopt` group
does. The golden schedule-name file gains it. No existing entry changes,
and `opensqp` joins no existing group, so no existing suite needs modOpt.

`opensqp` joins `PLAIN_RUN_OPTIMIZERS`. l2co_experiments' CUTEst
databank suite holds every plain-run entry, and
`tests/test_cutest_floor.py` there checks it. When l2co_experiments next
takes this package, either the suite adds `opensqp` and l2co_experiments
installs modOpt, or that test changes.
