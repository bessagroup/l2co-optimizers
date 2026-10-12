---
status: proposed
---

# Each registry entry declares how far it can enforce constraints

A task can declare constraints: one box, plus any number of inequalities
and equalities (l2co-tasks ADR 0002). Optimizers differ a lot in what
they can do with them, but nothing in the registry said so. A caller
building a suite for a bound-constrained or generally constrained task
set could not tell which optimizers can run it without reading every
factory.

**Decision.** Every registry entry has one of three levels.
`constraint_handling(name)` returns it, and `CONSTRAINT_HANDLING_LEVELS`
lists the levels weakest first:

| Level | Meaning |
|---|---|
| `none` | A box is applied only by clipping candidates. The optimizer's own state never learns that a point was moved. |
| `box` | The optimizer keeps its own state inside the box. |
| `general` | The solver also takes inequality and equality constraints. Implies `box`. |

## Clipping is not handling a box

Every built-in accepts the factories' `bounded` pair, and every point
any of them evaluates lies inside it. So "keeps its points in the box"
cannot be the line between `none` and `box`. The line is whether the
optimizer's *state* knows about the box. In most entries the clip comes
after the state has already moved:

- **optax and `lbfgs`** (`_src/optax_implementations.py`,
  `_src/lbfgs.py`). `optimizer.update` computes the step, and the new
  parameters are clipped afterwards. Adam's moments and L-BFGS's
  curvature pairs keep the unclipped step. On a face of the box, the
  moments go on pushing outwards and each step is cut back to the face.
- **evosax** (`_src/evosax_implementations.py`). The population is
  clipped after `ask`. The search distribution itself has no notion of
  the box and goes on proposing points outside it.
- **optimistix (`bfgs`, `dfp`, `nonlinearcg`, `neldermead`) and
  `trustkrylov`.** The loss is evaluated at the clipped point, but the
  solver's iterate, or Nelder-Mead's simplex, stays where its step put
  it. Outside the box the solver sees a loss that is flat along every
  clipped coordinate.

## Which entries are which

| Level | Entries | Why |
|---|---|---|
| `general` | `slsqp`, `trustconstr`, `cobyqa`, `cobyla` | scipy accepts `constraints=` for these four methods and no others (scipy 1.18.1, `scipy/optimize/_minimize.py`). COBYLA may step outside the box on the way; the driver evaluates its projection (ADR 0009). |
| `general` | `ipopt` | A nonlinear-programming solver; constraints would go in through casadi's `g`. |
| `box` | `lbfgsb`, `tnc`, `powell` | scipy takes `bounds=` for these methods itself. |
| `box` | `shade` | Its mutation moves a coordinate that leaves the box to the midpoint between the parent and the bound, so a population that starts inside the box stays inside it. |
| `box` | `turbo`, `rbftrustregion` | Candidates are clipped before the surrogate scores them, so the archive and the trust-region centre only ever hold points inside the box. |
| `box` | `randomsearch` | It has no state. Each draw is clipped and evaluated, and nothing carries over. Draws from a sampler range wider than the box pile up on its faces; that is accepted. |
| `none` | the other 63 | All optax and evosax entries, `lbfgs`, the four optimistix entries and `trustkrylov`, for the reasons above. |

## What a level does not say

- **`general` is not wired yet.** No entry receives an inequality or an
  equality today. l2co's `task_bounds` refuses a task that has one
  (l2co ADR 0020), and the scipy/IPOPT driver passes only a box.
  Passing them in also needs what l2co-tasks ADR 0002 lists: a "best so
  far" that prefers feasible points, and a `global_min` taken over the
  feasible region.
- **Nothing checks it.** l2co still runs every entry on a boxed task and
  clips as before. Refusing entries below `box` would today refuse 63 of
  75 on `gaussian_meta`, whose task has `Box(0, 1)`.
- **The parts path has no box.** `optimizer_parts` builds with
  `bounded=(None, None)`. SHADE, TuRBO and the RBF trust region run
  unbounded as sub-optimizers in a switching menu. A level describes
  the entry as its factory builds it.
- **One pair for every parameter.** The `box` and `general` solvers
  could take a different bound per parameter, but `bounded` carries one
  `(lo, hi)` pair (l2co ADR 0020).
- **Meta-optimizers have no level here.** They register with l2co, not
  in this registry.

## Where it lives

- The level names are string constants in `_src/core/constraint_handling.py`,
  like the `FAMILY_*` constants.
- `_src/mapping.py` holds one table of the built-ins above `none`. Any
  other entry is `none`.
- `register_optimizer` takes `constraint_handling=`, default `none`, so
  an external optimizer that only clips needs to say nothing.
- The README and docs optimizer tables gain a "Constraints" column.
  `tests/test_readme_optimizer_table.py` checks it against
  `constraint_handling`, so a new built-in gets a level when its row is
  written.

## Considered and rejected

- *Counting a clip as handling a box.* Every built-in would be `box`,
  and the level would say nothing.
- *A fourth level between `none` and `box` for "clips".* No built-in
  ignores `bounded`, so the bottom level would be empty.
- *A field on `UpdateClass`.* A caller choosing a suite by name would
  have to build every optimizer first.
- *Declaring the level next to each factory.* The level would be spread
  over ten modules. One table is easier to review, and the README
  column already sits next to each entry's description.
