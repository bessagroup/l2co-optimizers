---
status: proposed
---

# Plain-run entries that can be stepped join switching menus

A switching menu is the list of bare optimizers a meta-optimizer picks
from at every generation: l2co's strategy wrapper, rl2co's environment
and deployment wrappers, l2co_experiments' uniform-random floor. To be
on one, an optimizer needs three things:

- parts (`optimizer_parts`), the second construction path next to its
  `UpdateClass`;
- transfer ports, the reader and writer for the scale a switch carries;
- a handshake row in l2co's `HANDSHAKE_POLICY`, or the default.

Fifteen registry entries have none of this. Seven of them can get it.

## Which entries, and why not all fifteen

| Group | Entries | Can a loop step it one generation at a time? |
|---|---|---|
| JAX-steppable | `randomsearch`, `bfgs`, `dfp`, `nonlinearcg`, `neldermead` | Yes: their `init` / `step` already run inside a scan. |
| Reverse communication | `lbfgsb`, `slsqp` | Yes: scipy's Fortran routine returns to its caller after each evaluation, so its workspace can ride in the scan carry. |
| Solver-owned loop | `cobyla` (pyprima, pure Python in scipy 1.18), `cobyqa`, `powell`, `tnc`, `trustkrylov`, `trustconstr`, `ipopt`, and `opensqp` once it lands | No: the solver's state lives in Python locals or C++ and cannot be paused mid-run. |

The first two groups become switchable. The third stays plain-run for
now, and `PLAIN_RUN_OPTIMIZERS` keeps naming it. Three ways in were
weighed and none taken yet:

- **a restarted macro-step**: one generation runs a fresh solve from the
  incumbent for `k` evaluations. That gives a policy a different
  algorithm under the same name, and short visits are useless: COBYQA
  spends `2d + 1` evaluations building its first model.
- **deterministic replay**: carry the visit's evaluations in a buffer
  and replay them into a restarted solver each generation. Exact, but
  its cost grows with the square of the visit length.
- **reimplementing each solver in JAX**: these entries exist as named
  baselines of a specific library (ADRs 0002, 0003), so a
  reimplementation would defeat their purpose.

Host-side solver threads stay rejected: vmap and `while_loop` replays
corrupt state held outside the carry. Revisit the third group if a
menu needs one of them.

## One rule: `params` is the solver's next evaluation point

`GradientParts` and `PopulationParts` stay the only two kinds of parts.
l2co's `SubOpt` keeps its shape.

The five gradient solvers become `GradientParts` under one invariant:
**the `params` a loop holds is always the point the solver will
evaluate next.** The loop evaluates value and gradient at `params`, as
it does for every optax transform. `update` hands both to the solver
through a stub objective, a `jax.custom_jvp` that returns the loop's
value and gradient. It then returns `x_next - params`. One step is one
evaluation and bills 1. No `value_fn` is needed.

The loss value itself is needed, though, because a line search compares
losses. It arrives as `update`'s `value=` keyword, which optax's
line-search transforms already take. Until now a consumer forwarded it
only alongside a `value_fn`, and rl2co's `optax_wrapper` never did. So
every consumer now forwards `value` and `grad` on every gradient step:

- l2co's `strategy_wrapper` (l2co#86);
- rl2co's `evosax_wrapper` and `optax_wrapper` (rl2co#57).

All 27 optax transforms in the registry accept and ignore both.
`update` raises a `ValueError` naming the missing keyword when a
consumer does not forward it.

A third parts kind (`init` / `step(value_fn, ...)`, ADR 0001's sketch)
was rejected. It would need a new `SubOpt` branch in l2co and support
in both rl2co wrappers, for the sake of one solver (`neldermead`) that
can be rewritten instead (below).

`randomsearch` becomes `PopulationParts`: `ask` draws `popsize`
candidates from its sampler, and `tell` learns nothing.

## Per entry

**`randomsearch`.** `ask` uses the registry factory's sampler
(`random_search_sampling_fn`). It draws around the model, never around
the point a loop hands in, so a switch into it explores the task's
starting distribution, as a plain random-search run does. Its transfer
spec is unchanged: it reads nothing, writes nothing, and `own_ask`
stays false.

That leaves one generation where the two paths differ. On the
`UpdateClass` path the overridden `run` always draws its own
candidates. On the parts path a wrapper evaluates the handshake
population (`best_one`: the incumbent plus fresh draws) for the first
generation after a switch, then asks. Since random search holds no
state, this costs at most one re-evaluation of the incumbent.

**`bfgs`, `dfp`, `nonlinearcg`.** The registry `UpdateClass` stays as
it is (ADR 0001): the databank holds 467 `bfgs_linesearch=wolfe` cells
on `cutest`, and rl2co's environment already drives it. The parts share
its solver-step code, and only the objective differs: the real loss on
the registry path, the stub on the parts path. A test asserts the two
trajectories agree to a tight tolerance, not bitwise: XLA can fuse
`value_and_grad` and `linearize` plus transpose differently.

**`neldermead`.** It is rewritten as a JAX state machine that evaluates
exactly one point per generation:

- the first `n + 1` generations evaluate the initial simplex;
- after that, each generation evaluates the reflection, then, if the
  rules call for it, the expansion or a contraction;
- a shrink evaluates one shrunk vertex per generation.

`popsize` changes from `n + 1` to 1, and the family becomes
`derivative_free`. The entry now bills what textbook Nelder–Mead spends,
which removes ADR 0001's implementation tax. The initial simplex
follows scipy's rule: vertex `i` is `x0` with coordinate `i` scaled by
1.05, or set to 0.00025 when it is zero. The coefficients are the
standard (1, 2, ½, ½). The same code backs the registry entry, which
replaces optimistix's Nelder–Mead along with ADR 0001's correction to
it. No databank cell holds `neldermead`, so no data is stranded.

**`lbfgsb`, `slsqp`.** A stepped driver replaces ADR 0002's whole-run
driver under the same names and hashes (`_src/scipy_stepped.py`). It
keeps that driver's rules:

- a run lasts its whole budget;
- the final point is re-evaluated once scipy stops;
- a failure ends the run at the best point;
- a NaN loss is passed to scipy and never becomes the best.

How it is built:

- **One transform behind both paths.** Each method is one optax
  transform whose `params` is the point scipy evaluates next. The
  registry entry is an ordinary gradient `UpdateClass` around it: the
  loop evaluates the loss in the scan, at the point projected into the
  box with the gradient through the projection, as ADR 0002's driver
  did. `optimizer_parts` returns the same transform as `GradientParts`.
  So the two paths agree by construction, and `stop_fn` now works.
- **One host callback per step.** `update` feeds scipy the loss and
  gradient and runs its routine until it asks for the objective at a new
  point. A request at the point just evaluated -- L-BFGS-B's
  re-request after `START`, SLSQP's separate gradient request -- is
  answered from the values in hand, as scipy's own `ScalarFunction`
  cache answered it, so billing is unchanged. Under `vmap` the callback
  runs once per step for the whole batch (`vmap_method="expand_dims"`),
  looping over it on the host.
- **scipy's state rides in the carry as `uint32` bits.** The float64
  iterate and workspaces and the integer save arrays are viewed as
  `uint32` on the host and concatenated, in both x64 modes. The loop
  evaluates the float64 iterate cast to its own dtype, as ADR 0002's
  `jnp.asarray(x)` did. The callback returns only `uint32`: JAX checks a
  float64 callback output against the x64 setting of the thread the
  callback runs in, which a scoped `jax.enable_x64` does not reach.
- **Bounds** reach scipy natively on the registry path, so both entries
  stay `box` (ADR 0008). The parts path builds the transform unbounded
  (see "Bounds are deferred").

**The gate has two levels.** Level A replays the evaluations scipy's own
`minimize` loop makes into the transform, which must ask for exactly
the points that loop asks for, bitwise. Level B runs the entry end to
end against the whole-run driver, kept in `tests/whole_run_reference.py`
as the reference, on sphere, a 1e6-conditioned ellipsoid, Rosenbrock and
Rastrigin at d = 2, 10 and 40, plus a box, in float32 and in x64. For
`lbfgsb` both levels are bitwise.

**The run loop's sentinel shifts the history by one evaluation.**
`UpdateClass.run` runs `n + 1` steps and drops the first, so a
gradient entry's evaluation at x0 is never recorded or billed. The
whole-run driver overrode `run` and recorded x0 first. For the same
budget, the stepped history is therefore the whole-run one shifted by
one: `stepped[:-1] == whole_run[1:]`, bitwise, plus one more evaluation
at the end. This was accepted, so that `lbfgsb` follows the convention
of every other gradient entry. Overriding `run` to record the sentinel
was rejected: factories never subclass `UpdateClass` to change how a run
executes. The databank's `lbfgsb` cells no longer match a rerun entry
for entry and must be regenerated, or read shifted by one.

`_lbfgsb.setulb` and `_slsqplib.slsqp` are private scipy API. Level A
pins their behaviour at the installed version (scipy 1.18).

## Handshake: `best` + `reset` for all six stateful entries

Each of `bfgs`, `dfp`, `nonlinearcg`, `neldermead`, `lbfgsb` and
`slsqp` keeps its iterate inside its state, next to state that
describes that iterate's neighbourhood: a Hessian estimate, a
conjugate direction, a simplex and its losses, a Fortran workspace.

Under the default `continue`, a switch would hand the solver the
incumbent's loss and gradient while its state still points at the old
iterate. Nothing would raise. A `reset` re-`init`s it at the point the
loop steps from, which satisfies the invariant above by construction.

The rows live in l2co (l2co#86) and must merge before any entry here
becomes switchable. Keeping the Hessian estimate across a switch with
an iterate-moving writer was rejected for now: it pairs curvature with
a location it was not built at, which is the reason `lbfgs` resets.

## Transfer ports

| Entry | Family | Reader | Writer | `own_ask` |
|---|---|---|---|---|
| `bfgs`, `dfp`, `nonlinearcg` | gradient | RMS of `y_eval - y`, `CONF_ESTIMATED`; absent before the first step | none | no |
| `lbfgsb`, `slsqp` | gradient | RMS of `x - x_prev`, `CONF_ESTIMATED` | none | no |
| `neldermead` | `derivative_free` | RMS per-coordinate spread of the simplex, `CONF_ESTIMATED` | rescales the simplex so its RMS edge matches `sigma` | yes |
| `randomsearch` | population | none | none | no |

The gradient entries follow `lbfgs`: after a reset there is no slot a
carried scale survives in. The first step comes from H = I and Armijo's
`step_init`, or from inside Fortran. Scaling `bfgs`'s initial Hessian by
`sigma / ‖g₀‖` is a follow-up experiment, not part of this change.

`neldermead` sets `own_ask` because its next `tell` reads the phase and
pending vertex the handshake just wrote: the ADR 0014 hazard (l2co).
`derivative_free` gains a transfer spec for it.

## Bounds are deferred

The parts path has no bounds channel, and l2co's meta-optimizer clips
every population it is handed (l2co ADR 0020). On a bounded task that
breaks the invariant: the wrapper evaluates the clipped point, and the
solver is told an unmasked gradient. The registry path projects inside
the objective instead (ADR 0001).

No task carries a box today: CUTEst is restricted to unconstrained
problems. So the cross-path tests cover unbounded problems only, and on
a bounded task the parts path is a known divergence. Settle it with
the per-entry constraint declarations of ADR 0008.

## Rollout

1. l2co#86: the six handshake rows, and `strategy_wrapper` forwarding
   `value`; rl2co#57, which makes its two wrappers forward it.
2. This ADR and `randomsearch`'s parts.
3. The optimistix trio's parts.
4. The stepped `lbfgsb`, then `slsqp` on the same machinery, each behind
   its two-level gate. These went before the `neldermead` rewrite:
   nothing in them depends on it.
5. The `neldermead` rewrite.

Each step removes its own entries from `PLAIN_RUN_OPTIMIZERS` and
amends the "plain runs only" lines of ADRs 0001, 0002, 0003 and 0006.
