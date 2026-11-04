---
status: proposed
---

# COBYLA runs on the scipy host-callback driver

COBYLA (Powell, 1994) joins the registry as `cobyla`
(`_src/scipy_implementations.py`). It is a derivative-free trust-region
method: around its current point it fits linear models of the loss and
of every constraint, through a simplex of `d + 1` points. Like the other
scipy entries it is a plain-run entry (ADR 0002). It can never be a
sub-optimizer in a switching menu or an rl2co action.

## This reverses one row of ADR 0002

ADR 0002 left COBYLA out as "COBYQA's idea with a linear model". On a
problem without constraints that still holds, and more strongly than
the row suggests. The table gives the best loss in 5,000 evaluations,
or fewer if the method stopped, one run each from the same start, in
float64:

| Problem | d | `cobyla` | `cobyqa` |
|---|---|---|---|
| Sphere | 2 / 10 / 40 | 2e-31 / 6e-31 / 3e-30 | 2e-29 / 1e-25 / 1e-26 |
| Ellipsoid, condition 1e6 | 2 / 10 / 40 | 1.37 / 65 / 996 | 1e-17 / 1e-15 / 1e-15 |
| Rosenbrock | 2 / 10 / 40 | 4e-5 / 0.28 / 0.044 | 3e-18 / 9e-15 / 6e-8 |
| Rastrigin | 2 / 10 / 40 | 1.99 / 6.96 / 33.8 | 0 / 0.995 / 25.9 |

COBYQA also gets there in fewer evaluations: 51 against 129 on the 2-D
sphere, 202 against 3,137 at d = 40. A linear model cannot see
curvature, so COBYLA crawls along an ill-conditioned valley.

It is added anyway, for two reasons:

- **It takes constraints without gradients.** scipy accepts
  inequality and equality constraints for COBYLA. COBYQA is the only
  other derivative-free entry that takes them. A suite for a
  constrained task set needs more than one such method.
- **It is a named baseline.** modOpt lists it among its "performant
  algorithms", and engineering-design studies run it under its own
  name. ADR 0003 gave the same reason for SLSQP and trust-constr, and
  ADR 0006 for L-BFGS-B.

Under ADR 0008's constraint-handling levels (PR #35) it is `general`.
Like every entry at that level, it gets only the box today: l2co
refuses a task with inequalities or equalities (l2co ADR 0020).

## What scipy 1.18 runs

Since scipy 1.16, `minimize(method="COBYLA")` runs PRIMA's pure-Python
port of Powell's method (`scipy/_external/pyprima`), not the old Fortran
code. It takes bounds, inequalities and equalities. modOpt's wrapper
refuses an equality constraint, but that limit came from older scipy:
with 1.18.1, minimising `(x − 2)² + (y − 1)²` subject to `x + y = 1`
converges to (1, 0) in 24 evaluations with zero violation.

## Same driver, same rules

`cobyla` uses ADR 0002's driver unchanged, with the same rules as
`cobyqa`:

- one evaluation (a value) is one history entry, billed 1;
- a run lasts its whole budget, re-evaluating scipy's final point once
  it stops;
- failure ends the run at the best point;
- noisy and minibatched losses pass straight through;
- a `stop_fn` raises;
- the family is `FAMILY_DERIVATIVE_FREE`.

## COBYLA leaves the box on the way

The other scipy entries that take a box never evaluate outside it.
COBYLA does. PRIMA turns the bounds into linear constraints, `x ≥ lo`
and `x ≤ hi`. COBYLA does not keep its points feasible: it trades the
loss off against constraint violation, and may step across any
constraint, the box included, on its way to the optimum.

So the driver treats COBYLA the way ADR 0002 treats trust-krylov. It
evaluates the loss at the point projected into the box, and the history
records that projection. Every point in the history lies in the box.
COBYLA's linear model of the loss then sees a function that is flat
outside the box, while its model of the bound constraints pulls it back
in.

It still ends inside the box. On the twelve problems above, with the
optimum moved to 0.5 and the box `[−1, 0.2]` cutting it off, every run
ended on its trust-region radius. On the sphere, the ellipsoid and
Rosenbrock it ended at the loss `lbfgsb` reaches from the same start
while staying inside the box (on the 40-D ellipsoid, 0.02% above it).
On Rastrigin both end at local minima on the boundary, and from d = 10
not the same ones. How often COBYLA stepped outside on the way:

| Problem | d = 2 | d = 10 | d = 40 |
|---|---|---|---|
| Sphere | 11 of 24 | 99 of 130 | 476 of 704 |
| Rastrigin | 15 of 65 | 158 of 409 | 1,013 of 1,968 |

The alternative, evaluating the loss outside the box, would break the
rule that every evaluated point lies in `bounded`. A task's loss may be
undefined there.

## Tolerance: `tol = 1e-15`

`tol` is COBYLA's final trust-region radius, chosen as in ADRs 0002 and
0003, so a run ends only when COBYLA cannot progress. PRIMA does not
accept zero: it replaces it with 1e-6, with only a warning. At 1e-300
COBYLA raised `LinAlgError` ("Singular matrix") on a 3-D sphere. At
1e-15, on the twelve problems with a 5,000-evaluation budget:

| How it ends | Runs |
|---|---|
| Trust-region radius reaches `tol` | sphere and Rastrigin at every d; all twelve with the box |
| Runs the budget | the ellipsoid and Rosenbrock at every d |

None of these runs printed a warning. On another test function, with a
box, PRIMA printed numpy overflow and divide-by-zero warnings near the
end and still stopped on its radius. `tol = 1e-12` and `1e-10` end the
same runs the same way, sooner and less accurately: on the 40-D sphere,
after 2,469 and 2,106 evaluations instead of 3,137, at 3e-24 and 2e-20
instead of 3e-30.

## Hyperparameters

- `rhobeg`, the initial trust-region radius, at scipy's default of 1.0.
  The factory refuses a value that is not finite or is below `tol`.
  PRIMA would replace it, with only a warning, and the run would not be
  the one configured.

Not exposed:

- `catol`, the constraint tolerance. It matters only once constraints
  reach the method.
- `f_target`. It is an early stop, and a run lasts its whole budget.

The tolerance and the evaluation cap are fixed and hidden, as for every
scipy entry.

## Cost

Measured on one core with single-threaded BLAS and a cheap NumPy loss
(Rosenbrock). The figures are milliseconds per evaluation after the
`d + 1` evaluations of the starting simplex, which are cheap (0.2 to 0.4
ms each):

| d | 40 | 256 | 1024 | 3843 |
|---|---|---|---|---|
| COBYLA, ms per evaluation | 2.9 | 16.9 | 234 | 6,500 |

A databank run step allows about 14 ms per evaluation (ADR 0002). So
from about d = 256 upward, a cell that includes `cobyla` will hit the
wall-clock limit, as COBYQA's does from about d = 64. There is no
dimension cap, as for COBYQA.

PRIMA also keeps every evaluated point in a history list, since its
history length defaults to the evaluation cap. Memory grows by `8·d`
bytes per evaluation: at most 0.8 GB for 25,000 evaluations at
d = 3843, a run that cannot finish in time anyway. Peak memory was 1.2
GB at d = 3843 after 4,305 evaluations. scipy does not pass PRIMA's
`maxhist` option through.

## Config group

The `scipy` group gains `cobyla`, last. The golden schedule-name file
gains its entry. No existing entry changes.

`cobyla` joins `SCIPY_OPTIMIZERS` and so `PLAIN_RUN_OPTIMIZERS`.
l2co_experiments' CUTEst databank suite holds every plain-run entry,
and a test there checks it. That suite has to add `cobyla`, or the test
has to change, when l2co_experiments next takes this package.

## Rejected

- *Calling PRIMA directly*, through `scipy._external.pyprima`, to set
  `maxhist` and keep memory flat. That is a private scipy module. The
  memory only matters for runs that time out first.
- *Adding PRIMA's `prima` package* for its Fortran COBYLA. It would be a
  new dependency, and its iterates would differ from those of the
  `minimize(method="COBYLA")` that studies run.
