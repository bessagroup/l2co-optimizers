---
status: proposed
---

# Exact Hessians for `ipopt` and `trustconstr`

`ipopt` and `trustconstr` run with quasi-Newton Hessians (ADR 0003).
`ipopt` uses IPOPT's own limited-memory approximation; `trustconstr`
uses scipy's dense BFGS. Both solvers can also use the loss's exact
Hessian. modOpt's CUTEst benchmark runs them that way, as "IPOPT-2" and
"TrustConstr-2", and IPOPT-2 was its best performer (issue #28). ADR 0003
left exact Hessians out because of their cost.

**Decision.** Both entries take a `hessian` hyperparameter:

| Entry | Default | Exact |
|---|---|---|
| `ipopt` | `"limited-memory"` | `"exact"` |
| `trustconstr` | `"bfgs"` | `"exact"` |

The defaults are unchanged, so existing runs, schedule names and hashes
don't move. A config that writes `hessian: exact` gets its own name
(`ipopt_hessian=exact`). `limited_memory_max_history` now defaults to
`None`, which still means 6. It is refused with `"exact"`, which doesn't
use it.

## Where the Hessian comes from

The Hessian is `jax.hessian` of the loss, taken through the projection
into the box, as the gradient is. The driver serves it from
`_Driver.hessian(x)`, which calls a second jitted evaluator built next to
the value-and-gradient one (`_make_hessian_evaluator`).

- **casadi** gets the Hessian whole, as a callback passed in `nlpsol`'s
  `hess_lag` option. With no constraints, the Lagrangian Hessian is
  `lam_f · H`, and the callback returns its upper triangle. IPOPT asks
  for it once per iteration.
  - Rejected: letting casadi derive that function, by giving the
    gradient callback a Jacobian. casadi then spends most of each run
    building the solver: 1.4 s at d = 256 and 19.6 s at d = 512, before
    the first evaluation. With `hess_lag` the build takes 0.01 s at
    every d tried, up to 1024.
- **trust-constr** gets `hess=driver.hessian`. scipy calls it at `x0` and
  at each new iterate.

A loss computed outside JAX can be differentiated twice only if it
supplies its own Hessian. l2co-tasks ADR 0005 adds that to host
objectives, and the CUTEst adapter supplies pycutest's `hess`.

A loss that cannot be differentiated twice is refused before the method
starts. `_drive` traces the Hessian evaluator with `jax.eval_shape` and
raises a `ValueError` that says why. Inside the method, the same error
would have been taken for a failed run, and the run would have quietly
stayed at its starting point.

## A Hessian is one evaluation

Each Hessian the solver asks for is **one history entry, billed one
evaluation**. It records the loss at its point, recomputed on the same
sample. This was decided against two alternatives:

- **`d` evaluations:** about what `jax.hessian` costs in gradients.
- **Free:** modOpt's data profiles count objective evaluations only.

The consequence: on the evaluation axis, exact-Hessian runs get each
Hessian for the price of one evaluation, whatever `d` is, so they are
favoured more as `d` grows. On CUTEst an analytic Hessian costs about as
much as a gradient; from JAX it costs about `d` gradients. Wall-clock is
the real cost; see below.

## A Hessian is taken on its point's sample

On a noisy or minibatched loss, each evaluation draws its own batch and
key (ADR 0002). A Hessian is taken on the batch and key its point was
last evaluated with, so value, gradient and Hessian describe one
function, and the solver's local quadratic model is consistent. The
driver keeps the sample of its last 8 evaluated points. At a point not
evaluated lately, the Hessian is a fresh evaluation that draws the next
sample. A Hessian that reuses a sample doesn't advance the key stream.

## Where scipy and IPOPT stop

The failure rules are ADR 0002's.

- When the budget runs out in a Hessian request, the run ends as it
  would in a value request.
- After a failure or a spent budget, IPOPT's Hessian requests return NaN,
  as its value and gradient requests already do (ADR 0003).
- `tests/test_exact_hessian.py` ends runs with budgets of 1, 2, 3 and 7
  evaluations, all cleanly.

## Measured

Each Hessian is billed one evaluation. Evaluations to reach a loss of
1e-10 on an ellipsoid of condition 1e6, 10-D, in float64:

| | `ipopt` | `trustconstr` |
|---|---|---|
| Quasi-Newton default | not within 600 | 94 |
| `hessian="exact"` | 3 | 47 |

IPOPT takes one Newton step: a value and gradient, a Hessian, then the
minimum. On CUTEst, through l2co-tasks ADR 0005, the evaluations to
reach `f* + 1e-8` were:

| Problem | `ipopt` | `ipopt`, exact | `trustconstr` | `trustconstr`, exact |
|---|---|---|---|---|
| ROSENBR | 65 | 52 | 60 | 69 |
| BEALE | 15 | 18 | 16 | 21 |
| BROWNBS | 124 | 13 | 40 | 61 |
| HELIX | 53 | 30 | 30 | 31 |
| ARWHEAD, N = 100 | 16 | 11 | 12 | 13 |

Exact Hessians help most on badly scaled problems (BROWNBS). On easy
problems, trust-constr's billed Hessians can cost it more than they save.
A Newton method can also reach a different minimum: from Rosenbrock's
alternating 10-D start, exact `ipopt` stops at the local minimum
f = 3.987.

## Cost

Measured on one realization with a cheap loss, CPU. The figures are
milliseconds per evaluation, averaged over a run of 200 evaluations (20
at d = 3843), warm:

| d | 256 | 1024 | 3843 |
|---|---|---|---|
| `ipopt` | 2.6 | 3.6 | 32 |
| `ipopt`, exact | 4.8 | 40 | 997 |
| `trustconstr` | 1.0 | 2.0 | — |
| `trustconstr`, exact | 1.1 | 1.3 | 47 |

On this loss trust-constr converges within a few evaluations and spends
the rest of the run re-evaluating its final point, so its averages
understate what an iteration costs.

IPOPT's dense factorisation dominates. One `jax.hessian` of this loss
takes 0.23 ms at d = 1024 and 4.8 ms at d = 3843, where the matrix alone
is 118 MB in float64. Above roughly d = 500, exact `ipopt` exceeds the
~14 ms per evaluation a databank run step allows (ADR 0002). At
d = 3843 a 25,000-evaluation cell would take about 7 hours.

There is no dimension cap, as for COBYQA and SLSQP (ADRs 0002 and 0003).
A cell at large `d` can exceed the wall-clock limit and write nothing.

## Not done

- **Hessian-vector products.** trust-constr could take them (`hessp`)
  instead of a dense Hessian. Only the dense form is offered, which is
  the form modOpt's runs used.
- **No shipped config uses `"exact"`.** Adding one to the `scipy` or
  `ipopt` group adds a databank cell to every experiment that uses the
  group. That is an experiment-design decision.
- **Sparse Hessians.** IPOPT could exploit sparsity, but the Hessian is
  passed dense.
