---
status: proposed
---

# L-BFGS-B runs on the scipy host-callback driver

L-BFGS-B, the limited-memory BFGS code of Byrd, Lu, Nocedal and Zhu that
scipy ships as `minimize(method="L-BFGS-B")`, joins the registry as
`lbfgsb` (`_src/scipy_implementations.py`). It is a plain-run entry, like
the ADR 0002 and ADR 0003 entries: it can never be a sub-optimizer in a
switching menu or an rl2co action.

**This reverses one row of ADR 0002's "Not added" table.** ADR 0002 left
L-BFGS-B out as another implementation of `lbfgs`. On unconstrained
problems the two do behave alike (issue #26). The table gives the
evaluations to reach a loss of 1e-8, one run each from the same start,
in float64:

| Problem | `lbfgs` | `lbfgsb` |
|---|---|---|
| Rosenbrock 2-D | 42 | 43 |
| Rosenbrock 10-D | 93 | 86 |
| Rosenbrock 30-D | 204 | 205 |
| Ellipsoid, condition 1e6, 10-D | 452 | 596 |

It is added as a named baseline, for the reason ADR 0003 gives for SLSQP
and trust-constr: engineering-design studies run it under its own name.
modOpt's CUTEst benchmark, for one, includes it as "LBFGSB". A databank
comparison with such a study needs the method itself, not a stand-in.
It also differs from `lbfgs` in two ways that matter once a task has a
box:

- **Bounds.** L-BFGS-B handles the box inside the method. It moves along
  the projected gradient to a generalized Cauchy point, then minimises
  over the variables not held at a bound, and its limited-memory model
  knows which bounds are active. `lbfgs` takes an unconstrained step and
  clips the result into the box. Neither its line search nor its memory
  sees the clipping.
- **Line search.** Both accept strong-Wolfe steps with curvature
  constant 0.9, but through different codes. L-BFGS-B uses Moré–Thuente
  with `ftol = 1e-3`; `lbfgs` uses optax's zoom search with
  `slope_rtol = 1e-4`.

`lbfgs` stays as it is.

## Same driver, same rules

`lbfgsb` uses ADR 0002's driver unchanged, with the same rules as
`slsqp`:

- one evaluation (value and gradient together) is one history entry,
  billed 1;
- a run lasts its whole budget, re-evaluating scipy's final point once
  it stops;
- failure ends the run at the best point;
- noisy and minibatched losses pass straight through;
- a `stop_fn` raises;
- the box is passed natively, with the starting point clipped into it,
  and L-BFGS-B never evaluates outside it.

**Tolerances: `ftol = 0`, `gtol = 0`,** chosen as in ADRs 0002 and 0003.
The method ran on sphere, a 1e6-conditioned ellipsoid, Rosenbrock and
Rastrigin, at d = 2, 10 and 40, with a 5,000-evaluation budget:

| How it ends | Runs |
|---|---|
| Projected gradient exactly zero, or relative reduction exactly zero | sphere at d = 2 and 10, the ellipsoid at d = 2, Rosenbrock at d = 2 and 10, Rastrigin at every d |
| Line search fails ("ABNORMAL"), at f = 1.9e-29 | Rosenbrock at d = 40 |
| Proposes a non-finite point at the optimum | sphere at d = 40 |
| Runs the budget | the ellipsoid at d = 10 (reaches 5e-161) and d = 40 (4e-9) |

On sphere at d = 40 the iterate reaches f = 0. The next update then
divides by `sᵀy = 0`, and scipy 1.18's L-BFGS-B proposes non-finite
points. The driver never evaluates such a point: it treats the proposal
as a failure (ADR 0002) and ends the run at its best point, f = 0.
trust-krylov does the same at a machine-precision optimum.

L-BFGS-B never iterated without asking for an evaluation, so it needs
none of ADR 0003's stall guard.

**Hyperparameters** are scipy's scalar algorithmic settings, at scipy's
defaults:

- `maxcor`, the correction pairs kept (10);
- `maxls`, the line-search evaluations per iteration (20).

The factory refuses a value below one. scipy would raise only once the
run had started, inside the callback, and the driver would take that for
a failed run that never left its starting point. The tolerances and the
budget options are fixed and hidden, as for every scipy entry.

## Cost

Measured on one core with single-threaded BLAS and a cheap NumPy loss.
The figures are milliseconds per evaluation, including the method's own
overhead:

| d | 40 | 256 | 1024 | 3843 |
|---|---|---|---|---|
| L-BFGS-B | 0.06 | 0.07 | 0.13 | 0.32 |

That is far inside the roughly 14 ms per evaluation a databank run step
allows. Unlike SLSQP and trust-constr, its memory and cost per step grow
linearly with `d`.

## Config group

The `scipy` group gains `lbfgsb`. The golden schedule-name file gains
its entry. No existing entry changes.

## Rejected: stepping it from the JAX scan

scipy's L-BFGS-B hands control back after every evaluation
(`scipy.optimize._lbfgsb.setulb`), so its workspace could ride in the
scan carry, one evaluation per step (ADR 0002). That would be more work.
A switching menu would still need a new kind of parts, because the
method chooses its own trial points. The whole-run entry is enough for a
named baseline.
