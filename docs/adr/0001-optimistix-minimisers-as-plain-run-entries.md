# optimistix minimisers are plain-run registry entries

Four optimistix minimisers join the registry under bare names: `bfgs`,
`dfp`, `nonlinearcg` and `neldermead` (`_src/optimistix_implementations.py`).
They are the methods optimistix has and the registry lacked. optimistix's
`LBFGS` is left out on purpose: a second L-BFGS with a different line search
would make `lbfgs` comparisons ambiguous. `GradientDescent`, `OptaxMinimiser`
and `BestSoFarMinimiser` duplicate entries that exist. The least-squares
solvers (Gauss–Newton, Levenberg–Marquardt, dogleg) need residuals, and a
loss returns a scalar. `optimistix>=0.1.0,<0.2` is a hard dependency; it adds
only `lineax`. The cap is there because a solver's `init` / `step` signature
is the surface we drive, and optimistix is pre-1.0.

**One optimistix step is one l2co step.** Each factory builds the solver once
and drives its `init` and `step` directly from `init_fn` / `step_fn`. Nothing
calls `optx.minimise`, which would run its own loop to convergence. **Nothing
calls `solver.terminate` either.** `rtol` / `atol` are fixed constants that
affect nothing. A run ends at its budget or when `stop_fn` fires, like every
other entry. If a solver stopped itself, its trajectory would be NaN-padded,
and the ERT would score a converged-and-idle solver as censored.

**Billing counts the calls optimistix makes.** A gradient solver's step is one
`jax.linearize` of the loss, so it bills 1. Nelder–Mead bills `n + 1` on its
first step, when it evaluates the injected simplex, and 2 on every later step.
A shrink step bills `n + 3` (`max_fevals_per_step`). That is more than the
algorithm needs. optimistix evaluates the reflection and then the
expand/contract vertex in a two-iteration scan, even when the reflection is
simply accepted, and a shrink re-evaluates the best vertex too. SciPy's
`nfev` for the same run is roughly half. We bill what is spent, as l2co ADR
0004 requires. `neldermead` therefore pays an implementation tax, and it stays
out of cross-optimizer comparisons in papers. A test counts the real loss
calls with a host callback and asserts they equal the billed `fevals`, both
eager and jitted.

**The history records what was evaluated.** Each step of a gradient solver
evaluates the trial point `y_eval` that the previous step chose. That point
may be a line-search trial that gets rejected. It is recorded with its loss,
because a rejected trial can still lower the best-so-far. optimistix keeps the
loss only on acceptance. A `RecordingSearch` wraps the Armijo search through
optimistix's public `AbstractSearch` extension point and saves `f_eval` and
the accept flag in its own state, so the value costs no second evaluation.
The gradient is recorded when the step was accepted and is NaN otherwise:
optimistix forms it only on acceptance, and computing it ourselves would be
hidden work. Nelder–Mead records the post-step simplex, its losses and NaN
gradients. NaN gradients are already how population entries record "not
computed", and the grad-norm stop uses `nanmean`.

**Bounds are a projection inside the objective.** optimistix keeps its iterate
(and its Hessian estimate or simplex) in its own state, so clipping the
returned `y` would desynchronise it. The objective evaluates the loss at
`clip(y)`, and the history and the population l2co carries are projected too.
The solver's state keeps the unprojected iterate, in a `GradientSolverState`
for the gradient solvers. Outside the box the gradient along clipped
coordinates is zero, so a gradient solver can slide along the boundary.

**Stochastic and minibatched losses pass straight through.** Each step gets
the fresh batch and key, like every other entry. The Armijo test compares the
new loss against the loss cached from the previous step. On such a loss those
are two different functions, so the line search is noisy. That is accepted,
not corrected. Freezing the sample until a step is accepted would need the
optimizer state to carry the batch and would bend `step_fn`'s contract. No
branch depends on `pass_rng`. Within one Nelder–Mead step, every evaluation
shares the step's key.

**The simplex is the population.** `neldermead` has `popsize = n + 1`,
derived from the model and not a hyperparameter, and family `population`. The
sampler's initial population is the initial simplex. optimistix 0.1.0 cannot
build that simplex itself:

- its `y0_simplex=True` option rejects every simplex with n > 1, because it
  sums `x[1:].size`, which is n²;
- the simplex it grows from one point by default is degenerate for n >= 3. It
  perturbs flat positions `1..n` of the `(n + 1, n)` array, so rows `2..n`
  all equal the start point.

So `init_fn` builds the default state from one vertex and swaps the
population in with `eqx.tree_at`. `f_simplex` is still all-inf and
`first_pass` is still true, so the first step evaluates the injected simplex
without moving it.

**One Nelder–Mead correction.** optimistix 0.1.0 computes the best and worst
indices from the updated simplex, but reads the vectors at those indices from
the pre-update one (`nelder_mead.py`, the `top_k` block at the end of `step`).
After a replacement or a shrink, `state.best` can hold the old worst vertex.
This happened on 91 of 300 steps on 3-D Rosenbrock. The next step then
compares against, and shrinks toward, the wrong point. From the same simplex,
5-D Rosenbrock stalls at f ≈ 3.9 where SciPy converges. After every step,
`_reindex_best_and_worst` re-reads both vectors from the updated simplex.
That is the state the indices and stored losses already describe. It
evaluates nothing, so billing is unchanged. With it, 5-D Rosenbrock reaches
1e-8 in 469 steps. This departs from replicating optimistix exactly.
Reproducing a known-wrong update under the name `neldermead` would serve no
one. `test_upstream_best_vector_goes_stale` fails once upstream is fixed, and
then the correction can go.

**Plain runs only.** None of the four can be unpacked by `optimizer_parts`.
An optimistix solver evaluates the objective itself. It is not an optax
`GradientTransformation`, which takes gradients and returns updates, and it
is not an ask/tell population. `optimizer_parts` raises a ValueError that
names this, and no handshake policy or transfer ports are defined. Putting one
in a switching menu would need a third parts kind (`init` / `step(fn)`) and a
matching SubOpt adapter in l2co. That is a separate decision.

**Realizations.** Under the fused driver the realization axis is vmapped,
and that turns optimistix's `lax.cond` into a select. For Nelder–Mead the
shrink branch (`n + 1` loss evaluations) then runs on every step.
`NelderMeadUpdateClass` sets `sequential_realizations = True`. Measured warm
`batch_run` walltime for an expensive loss, 8 realizations and 200
iterations: sequential is 3.0× faster at n = 10 and 6.6× faster at n = 40.
The gradient solvers stay fused. Their branches differ only by the
Hessian/β update, and fused is 1.65× faster at n = 10 and 1.08× at n = 40.
With a cheap loss both families favour fused below n ≈ 10, but those are
millisecond-scale overheads.

**Hyperparameters.** `bfgs` and `dfp` take `use_inverse` and the Armijo
`decrease_factor`, `slope` and `step_init`. The stock solvers hard-wire
`BacktrackingArmijo()`, so the factories swap in the configured search with
`eqx.tree_at`. `nonlinearcg` takes `method` (`"polak_ribiere"`,
`"fletcher_reeves"`, `"hestenes_stiefel"` or `"dai_yuan"`) and the same
Armijo knobs. `neldermead` takes none. All are scalars or strings, so
`OptimizationStep` hashes stay simple. The `optimistix` config group ships
all four at their defaults.
