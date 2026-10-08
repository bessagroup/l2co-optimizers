---
status: proposed
---

# A strong-Wolfe line search for `bfgs`

`bfgs` is optimistix's BFGS with a backtracking Armijo line search
(ADR 0001). When a published comparison says BFGS, it usually means
scipy's `minimize(method="BFGS")`, which uses the same inverse-Hessian
update but a line search that enforces the strong Wolfe conditions. On
ill-conditioned problems that difference dominates the cost (issue #27).
On an ellipsoid of condition 1e6, `bfgs` needed 122 evaluations (10-D)
and 473 (40-D) to reach a loss of 1e-8; scipy needed 26 and 63. At least
68% and 83% of `bfgs`'s evaluations were rejected line-search trials.

**Decision.** `bfgs` gains a `linesearch` hyperparameter:

- **`"armijo"`**, the default. This is what `bfgs` has always been.
- **`"wolfe"`**: scipy's BFGS, made steppable. It is `WolfeBFGS`
  (`_src/optimistix_implementations.py`), driving a JAX port of scipy's
  Moré–Thuente line search (`_src/more_thuente.py`).

In float64, `"wolfe"` evaluates the points scipy's BFGS evaluates, up
to the point where scipy stops (to 1e-6 relative in
`tests/test_bfgs_wolfe.py`). On the ellipsoids above it needs exactly
scipy's 26 and 63 evaluations.

## Why a port of scipy's search

scipy's BFGS searches with `line_search_wolfe1`, which is MINPACK-2's
`dcsrch`. `dcsrch` is reverse-communication by design: each call takes
the loss and slope at one trial step and returns the next trial step.
That fits ADR 0001's rule of one solver step per l2co step, one
evaluation each. `more_thuente.start` and `more_thuente.iterate` port it
branch for branch, with `jnp.where` instead of `if`, so the port runs
under `jit` and `vmap`. `tests/test_more_thuente.py` drives scipy's own
`DCSRCH` and the port with the same values. They name the same trial
steps to 1e-12, and they converge, warn or refuse to start in the same
cases.

**The constants are scipy's.**

- `c1 = 1e-4`, `c2 = 0.9`, `xtol = 1e-14`.
- Steps are bounded to `[1e-100, 1e100]`. In float32 those overflow, so
  the bounds narrow to the square roots of the smallest normal and the
  largest finite number.
- A search may take 100 evaluations.
- The first trial step is `min(1, 2.02 (f_k − f_{k−1}) / φ'(0))`, with
  `f_{−1} = f_0 + ‖g_0‖ / 2` on the first search.

None of these is a hyperparameter: `"wolfe"` means scipy's search.

Rejected: optax's zoom search, which `lbfgs` uses. It runs its own
`while_loop` and calls the loss itself, so it cannot be stepped one
evaluation at a time, and it is not the search scipy's BFGS uses.

## Why `step` is overridden

optimistix gives a search only the loss at the point it just evaluated
(`FunctionInfo.Eval`), and forms the gradient there only when the step
is accepted. The strong Wolfe conditions also need the slope at every
trial. So `WolfeBFGS` subclasses `optx.AbstractBFGS` and overrides
`step`.

- Each step evaluates value and gradient together (`jax.value_and_grad`).
- It takes the slope along the search direction from `NewtonDescent`'s
  state.
- It keeps optimistix's state layout. `_gradient_solver_update`,
  `init`, the descent and the Hessian operator are all reused unchanged.

It imports no private optimistix name. It does read or replace optimistix
0.1 internals: the solver state's fields, `descent_state.newton` and the
operator's `.pytree`. `test_upstream_internals_it_relies_on` pins them,
and the `optimistix<0.2` cap stands.

**The estimate is updated whenever `sᵀy > 0`.** scipy updates whenever
`sᵀy ≠ 0`. optimistix updates only when `sᵀy` exceeds the dtype's
epsilon. That absolute threshold is 1.2e-7 in float32, so optimistix
stops updating once steps shrink to about 1e-4, near any optimum.
Requiring a positive value keeps the estimate positive definite. After a
successful Wolfe search `sᵀy` is positive anyway, except through
rounding.

## Where scipy stops, a run goes on

A run lasts its budget (ADR 0001). scipy's BFGS stops when its line
search fails, after trying `line_search_wolfe2` as a fallback, which is
not ported. Here:

- **A failed search** — a warning from the search, 100 evaluations, or a
  non-finite next step — accepts its last point if that point lowered
  the loss. Otherwise the estimate resets to the identity and a new
  search starts from the accepted point.
- **A direction that does not go downhill** is replaced by steepest
  descent. If even that does not go downhill, because the gradient is
  zero or not finite, every step re-evaluates the accepted point. The
  scipy driver re-evaluates its final point in the same way (ADR 0002).
- **A non-finite loss or slope at a trial** halves the step towards the
  best step of the search so far, and does not update the search.

So a run is scipy's until scipy's first failed line search, and its own
after that. The tests run it past convergence, from a stationary start,
across a region where the loss is NaN, on a noisy loss and in float32.
None of them produces a NaN iterate.

## Billing, history, bounds, noise

- **Billing:** one evaluation per step, value and gradient together,
  billed 1 (`fevals_bound == 1`).
- **History:** the gradient is recorded at every evaluated point, since
  the search computes it anyway. The Armijo path still records NaN on
  rejected trials (ADR 0001).
- **Bounds:** the box is a projection inside the objective, as in
  ADR 0001.
- **Noise:** a noisy or minibatched loss makes the search compare values
  from different batches. This is accepted, as in ADR 0001. The failure
  policy above keeps such runs finite.

**Cost.** Measured on one realization, float32, 8 CPU cores, with a
cheap loss. Figures are milliseconds per step.

| d | 40 | 256 | 1024 | 3843 |
|---|---|---|---|---|
| `"armijo"` | 0.007 | 0.04 | 0.33 | 2.2 |
| `"wolfe"` | 0.034 | 0.12 | 0.41 | 1.7 |

At small `d` the search's fixed overhead shows. At large `d` the dense
operator dominates both. All of these are far below the roughly 14 ms
per evaluation that a databank run step allows (ADR 0002).

## Why a hyperparameter, not a new default

Changing what `bfgs` does would change the meaning of every stored `bfgs`
databank cell. A schedule's name and hash depend only on the
hyperparameters a config writes, so the existing names don't move, and
a config that writes `linesearch: wolfe` gets its own name
(`bfgs_linesearch=wolfe`).

The Armijo settings (`decrease_factor`, `slope`, `step_init`) now
default to `None`, which means 0.5, 0.1 and 1.0 as before. Giving any of
them with `"wolfe"` raises a `ValueError` instead of being silently
ignored.

## Not done

- `dfp` and `nonlinearcg` keep Armijo. The same override would serve
  `dfp`. `nonlinearcg` would need a different curvature constant
  (`c2 ≈ 0.1`).
- No shipped config uses `"wolfe"` yet. Adding one to the `optimistix`
  group would add a databank cell to every experiment that uses the
  group, which is an experiment-design choice.
- scipy's `line_search_wolfe2` fallback is replaced by the failure policy
  above.
