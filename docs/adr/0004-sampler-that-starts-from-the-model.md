---
status: proposed
---

# A sampler that starts from the model's own values

A sampler draws the starting population of every run. Today no task can
choose where that population starts. All five samplers (`random`,
`normal`, `xavier`, `constant`, `grid`) read the model's parameters only
for their shape (`_src/core/sampler.py:_sample`), so every run starts
near the origin of the task's own coordinates. bbob and the
neural-network tasks don't care. CUTEst problems do (l2co-tasks ADR
0004): each one prescribes a starting point `x0`, CUTEst comparisons
start there, and points far from it can overflow.

**Decision.** A new sampler, `relative_normal_sampling` (`get_sampler`
name `"relative_normal"`), reads the *values* of the parameters it is
given:

- **member 0 is exactly the given parameters;**
- **every other member is a relative Gaussian perturbation of them,**
  `x_i + σ · max(|x_i|, 1) · N(0, 1)` per coordinate, with `σ` a
  keyword that defaults to `1.0`.

A task that prescribes a start sets its `model` to that start (l2co-tasks
ADR 0004 does, for CUTEst). The sampler stays task-agnostic: it never
sees a task, only `sampler(key, params, n_samples)`. It lives in
`_src/core/sampler.py`, because it names no optimizer. It is public like
the other samplers, re-exported by `l2co.sampling` and added to
`tests/test_optimizers_facade.py`'s origin table. l2co_experiments gains
a `samplers/relative_normal.yaml`.

## Why member 0 is exact

Every optimizer with a population of one starts at `params[0]`: the
optax and optimistix entries, `lbfgs`, the scipy minimisers and IPOPT.
So does the mean of every distribution-based evosax strategy
(`evosax_implementations.py`, `init_fn`). Keeping member 0 exact means all
of them start at the prescribed point, and their results can be checked
against published CUTEst results.

The cost is accepted: a deterministic optimizer with a population of one
makes the same run in every realization. On CUTEst an evaluation takes
microseconds, so the repeated runs are cheap. The ERT handles identical
realizations correctly.

## Why the other members are spread, and relative

Population optimizers need spread. DE and SHADE mutate along differences
between members; a population sitting entirely at `x0` would never move.

The spread is relative to each coordinate's size because CUTEst
variables differ in scale by orders of magnitude, both within one
problem and between problems. An absolute spread of 1 swamps a variable
of size `1e-3` and is invisible on one of size `1e4`. The `max(|x_i|, 1)`
floor keeps components at or near zero from getting no spread at all.
That is common, since many `x0` have zero components. Scaling with `x0`
follows the CUTEst habit of making harder starts by scaling it
(Moré–Garbow–Hillstrom use `10·x0` and `100·x0`). With `σ = 1` and
variables no larger than 1, the spread matches the databank's current
`N(0, 1)` start, so DE and SHADE see a comparable initial spread to
bbob.

## When it is called after a switch

l2co and rl2co also call the experiment's sampler, with the incumbent
(the best point so far), to rebuild a population after a switch:
`handshake_policy.py` draws `popsize` points and overwrites the last
with the incumbent; `strategy_wrapper.py` and rl2co's `evosax_wrapper.py`
draw `popsize − 1` and place the incumbent at index 0 themselves. With
this sampler, two things follow:

1. the population holds the incumbent twice, which wastes one member;
2. the other members are drawn around the incumbent, not around the
   origin as with the current samplers, which ignore the values they get.

Both are accepted and left as they are. They only matter for a
meta-optimizer run on a task set that uses this sampler, and none is
planned. A flag that drops the exact member when the caller inserts the
incumbent itself was rejected: it would put knowledge of switching,
which is l2co's concern (l2co ADR 0019), into a sampler.

## Considered options

- **Shift inside the loss** (`model = zeros`, evaluate `f(x0 + z)`).
  The existing samplers would then centre on `x0` unchanged. Rejected:
  stored parameters would be offsets from `x0`, and a box would turn
  into one per variable.
- **Hand the sampler `x0` from the task** (for example through a tag).
  Rejected: it needs new plumbing in l2co to pass task data to a
  sampler, and nothing passes it today.
- **Absolute spread** (`x0 + σ·N(0, I)`). Rejected for the scale reason
  above.
- **A uniform box** of `±σ·max(|x0|, 1)`. Same scaling but bounded. Not
  needed while CUTEst tasks are unconstrained.
- **Perturb member 0 as well.** Every realization would differ, but no
  run would start at the prescribed point. Rejected.

## Also used by `estimate_global_min`

l2co-tasks' `estimate_global_min` gains a `restart_sampler` parameter
(l2co-tasks ADR 0004). l2co-tasks never imports this package, so the
caller (an l2co_experiments experiment) passes this sampler in.
