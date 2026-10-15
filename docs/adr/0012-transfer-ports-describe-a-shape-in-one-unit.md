---
status: proposed
---

# Transfer ports describe a shape in one unit, and gradient optimizers keep a curvature record

l2co ADR 0022 changes what crosses an optimizer switch. No scale
crosses any more. A receiver whose own state no longer fits where it is
sent is **stale** and **restarts**. A **shape** crosses only where it
can be trusted. A shape is how much more curved the landscape is along
some directions than along others, with the overall size removed.

The switching itself is l2co's. What each optimizer exposes to it is
this package's. So this package settles four things:

1. the unit and layout a shape is expressed in;
2. what every switchable optimizer must declare, so l2co can decide
   staleness and restarts for all of them;
3. how each optimizer reads a shape out of its state and writes one in;
4. where gradient optimizers that hold no shape of their own get one.

It covers every switchable entry in the registry, not just `headroom4`.
That is 66 of the 75 entries; the 9 in `PLAIN_RUN_OPTIMIZERS` cannot be
switched to. Nothing here is implemented yet.

## Each optimizer stores shape differently

Write h for the landscape's curvature along a direction. On a quadratic
bowl, each optimizer's field relates to h by a fixed power, but not the
same power:

| Kind | Fields | Relation to h |
|---|---|---|
| Area-like | CMA-ES `C`, Sep-CMA-ES `C`, CR-FM-NES `D(I + vvᵀ)D`, BFGS/DFP inverse Hessian, Nelder–Mead vertex spread | ∝ 1/h |
| Length-like | SNES `std` | ∝ 1/√h |
| Curvature | a stored Hessian rather than its inverse | ∝ h |
| Partial | L-BFGS's ≤10 secant pairs | exact 1/h, but only along directions it has stepped in |
| Mixed | Adam's `nu`, Rprop's `step_sizes` | ∝ h × distance still to go |

Reading a field with the wrong power squares or square-roots how
stretched the valley looks. A valley with a 1 000 000 : 1 ratio
arrives as 1 000 : 1, or as 10¹² : 1.

The mixed fields cannot be converted at all. Adam's state holds running
averages only, with no earlier points and no gradient differences. One
number per coordinate cannot be split into "how curved" and "how far".

## Decision

### One unit: the inverse-curvature shape

A shape is expressed in the area-like form above, the kind of number
CMA-ES's covariance holds. It is normalised so that its determinant is
1, which is the same as saying the geometric mean of its stretches is 1.
Three reasons:

- It is the native form of most providers, so most readers need no
  conversion.
- Normalising by the geometric mean survives a change of power: square
  or square-root a shape normalised this way and it stays normalised.
  A unit conversion therefore cannot change the size by accident.
- It carries no size, because under l2co ADR 0022 the receiver keeps
  its own scale.

### One layout: a diagonal and two signed directions

Every reader's output passes through one `lax.switch`, so the layout is
fixed:

```text
shape = D (I + Σᵢ (sᵢ − 1) uᵢuᵢᵀ) D,   with det = 1
```

- **D** is the diagonal.
- There are **k = 2 directions** uᵢ: the **widest** and the
  **narrowest** relative to the diagonal. They are orthonormal after
  scaling by the diagonal.
- Each stretch sᵢ is **relative to the diagonal**. Above 1 widens, below
  1 narrows.
- **Each part has its own presence flag and its own evidence**: whether
  the diagonal was measured, and whether each direction was found. A
  reader leaves a direction out unless its stretch is clearly away from
  1, by a factor the slope test fixes.

The size is 4d plus a few numbers.

This is CR-FM-NES's own form, extended by one direction. CR-FM-NES
samples x = mean + σ·D·y with y = z + (√(1+|v|²) − 1)(z·v̂)v̂, so its
covariance is σ²·D(I + vvᵀ)D. It normalises to det 1
(`evosax .../cr_fm_nes.py:182–183, 276–279`). Its `v` can only widen a
direction, so it takes the widest direction and nothing else. Six
receivers can hold a narrow direction (cmaes, iamalgamfull, maes, xnes,
bfgs, dfp), and for a narrow valley the narrow direction, the valley
wall, is the most valuable thing to hand over. That is why there are
two directions and not one.

**Negative curvature.** A reader that meets zero or negative curvature
floors it at 1e-6 × the largest curvature before inverting. "Not curved
upward" is treated as "very flat". The oracle experiment uses the same
floor (l2co ADR 0022).

**Writers keep the receiver's own size.** A writer multiplies the
det-1 shape by the receiver's own overall size (its geometric-mean
width) before writing it. The shape changes, the size does not.
Unmeasured parts are taken from whatever the receiver holds at that
point: its own shape, or its fresh one after a restart.

### What every switchable optimizer declares

Every switchable registry entry declares the same properties. One test
runs over the whole registry and checks that each property is present
or explicitly "none".

| Property | What it is | Defaults |
|---|---|---|
| **Scale reader** | the existing σ readers, now also the denominator of l2co's distance gate | as today |
| **Point reader** | where the optimizer is | ES: `mean`; population: centroid; TuRBO, RBF trust region: `center`; Nelder–Mead: the simplex centroid; gradient: the iterate, which the switching layer holds |
| **Shape reader and writer** | into and out of the layout above | may be "none" |
| **Evidence rate** | per shape part; see below | required for every shape reader |
| **Restart** | what a restart keeps; see below | `init` at the destination |

An optimizer whose scale reader returns "absent" needs no point reader.
l2co ADR 0022 counts it as stale on every switch with news, so it always
restarts.

#### Scale readers that feed a gate must be accurate

Until now a scale reader only fed the scale write. Under l2co ADR 0022
it decides whether a member keeps its state or restarts. A reader that
is 5× off moves the threshold by 5×.

The Adam-family reader (`make_read_sigma_gradient`) computes lr·m/√v
and skips bias correction. The size of the error is
(1 − β₁ᵗ)/√(1 − β₂ᵗ). With Adam's defaults that is 3.2× at step 1,
6.5× at step 10, 3.2× at step 100 and 1.26× at step 1000. Short visits
are the normal case under a switching policy.

- The Adam-family reader is **bias-corrected**. It reads the count from
  the state (it already looks up `count`/`t`) and closes over β₁ and β₂
  at construction, the same way it closes over the learning rate.
- **An accuracy test** checks every gradient scale reader against the
  step the optimizer actually takes next, from step 1 onwards. It must
  agree within a factor of 1.5. A reader that cannot meet that reports
  "absent", and its optimizer restarts on every switch with news.
- ES and population readers are exempt. Their σ or spread is the scale
  by definition.
- l2co ADR 0013's three confidence levels stay, as labels. A gate
  treats "estimated" like "exact" once the reader has passed the test.

#### Evidence rates

The evidence rate of a shape part is **the fraction of that part
replaced per update**, taken from the optimizer's own parameters. Where
several terms write into the same part, they are added. In CMA-ES, for
example, the matrix forgets at (1 − c1 − c_mu) per generation, so using
`c1` alone would understate how fast it adapts by a factor of about
μ_eff.

| Optimizer | Rate |
|---|---|
| cmaes, sepcmaes, maes | `c1 + c_mu` |
| snes | `lr_std` |
| xnes | `lr_B` |
| crfmnes | `lr_B` for `D`, `c1` for `v` |
| lmmaes | `c_d`, one per direction |
| iamalgamfull, iamalgamunivariate | `eta_std` |
| rmes | `c_mu`, `c_c` |
| simplees | `c_std` |
| pair-based readers | kept pairs; see below |

l2co counts evidence as updates × rate. A part passes at ≥ 1.

**No identifiable rate means no shape reader.** The optimizer declares
"none" until someone works out its rate. There is no generic fallback,
because a default such as 1/d is wrong by orders of magnitude for
anything that is not CMA-like. pgpe and des showed no rate field in a
first look, so they stay "none" until checked.

#### Restart

A restart is declared per optimizer in `TransferSpec`, next to
`own_ask`. One test checks that each declaration keeps what it says.

- **Default:** the optimizer's own `init` at the destination.
- **TuRBO and the RBF trust region:** keep the archive of evaluated
  points, move the centre to the destination, reset the length or
  radius to its initial value and zero the counters. A measured
  function value never goes stale, and this is the restart these two
  already do when their region collapses (`turbo.py:411–416`,
  `rbf_trust_region.py:469–473`). `init` would zero the archive
  (`turbo.py:300`). TuRBO's GP lengthscale is tied to the length
  (`turbo.py:164`), so it resets too, as a stale scale should.
- **SHADE and PSO** take the default. Their stored points drive the
  search rather than describe the function. SHADE's archive feeds
  difference vectors, so old far-away points would mean huge steps.
  PSO's personal bests pull particles back.

### Shapes from step pairs

L-BFGS's memory and the curvature record (below) both hold up to 10
pairs (s, y): a step and the change in gradient along it. One reader
serves both.

- **Kept pairs.** A pair is kept only if sᵀy > ε·|s|·|y|, meaning
  positive curvature along the step. This is L-BFGS's own safeguard,
  and it is the only defence against noise. A pair whose step is below
  1e-8·max(|x|, 1) is also dropped, because its gradient change is
  mostly rounding error.
- **Measurement scale:** the geometric mean of the kept pairs' RMS step
  lengths. It is *not* the σ reader. L-BFGS's σ reader is the RMS of
  its *latest* step (`state_transfer.py:598–618`), which goes to about
  1e-32 as it converges, while older pairs in the same memory were
  measured at useful scales.
- **Measurement point:** the centroid of the points the kept pairs
  connect. The points are rebuilt by stepping back from the latest point
  and subtracting the stored steps, so nothing extra is stored. This
  only works because a pair set always comes from one unbroken stretch
  of path (see "Cleared on news" below).
- **Diagonal:** fitted from the pairs, hⱼ = Σᵢ sᵢⱼyᵢⱼ / Σᵢ sᵢⱼ². It is
  flagged as measured only if every hⱼ is positive and the relative
  residual |y − diag(h)s| / |y| is below 0.5. On a separable function it
  is exact, even when every step points the same way, which is what Adam
  and Rprop do. On a rotated function the cross terms leak in, and the
  residual test switches it off. It needs at least 3 kept pairs.
- **Directions:** the pairs are scaled by the diagonal. The direction
  of least curvature is the widest, and the direction of most curvature
  the narrowest. Each is kept only if its stretch is clearly away from
  1. If the diagonal is not measured, a single best-fitting value takes
  its place, and the stretches are relative to that.

### The curvature record

An optimizer that computes a gradient at every point it visits is
already measuring curvature. The secant relation, y ≈ curvature × s, is
a property of the function, not of the optimizer that chose the step.
The record keeps what the optimizer would otherwise throw away.

- **Content.** The previous evaluation point and its gradient, plus a
  ring buffer of the last **10** pairs between **consecutive**
  evaluation points. It records points after box clipping, so the pairs
  describe steps actually taken. Line-search trials count: any two
  evaluated points give a valid pair.
- **Cleared on news.** On a switch with news, l2co clears the whole
  record, both the previous point and the pairs. A switch without news
  leaves it untouched, previous point included, because the optimizer
  resumes from where it was. A record therefore always describes one
  unbroken stretch of path.
- **Read-only.** The optimizer's own update never sees the record. This
  must be checked bit for bit: trajectories with and without the record
  must be identical.
- **A construction keyword, off by default.** Factories and
  `optimizer_parts` take `curvature_record=`, next to `bounded=`. It is
  not a hyperparameter, so schedule names and hashes do not change. Only
  consumers that can switch turn it on:
  - rl2co's environment, through `l2co.init_run_state`;
  - the deployment wrappers, through `optimizer_parts`.

  Static runs never carry it: no extra memory in the databank, and no
  new way for stored rows to break. `tests/test_optimizer_parts.py`
  keeps the two construction paths in step.
- **Who has one.** Every gradient optimizer **except L-BFGS**, whose
  memory already is a record. That includes BFGS and DFP. They hold an
  inverse Hessian built from every step since their last restart, but
  store no step history, so their scale and point would otherwise be
  unknowable. Their σ reader is the *upcoming* trial step
  (`state_transfer.py:639–670`). So BFGS and DFP read their shape from
  the record like everyone else. The stepped solvers (bfgs, dfp,
  nonlinearcg, lbfgsb; ADR 0011) present their next evaluation point as
  `params`, so the record sees what it needs.
- **Every task.** The record is on for every task when switched on. It
  is never chosen by noise or task type, because whether a loss is
  deterministic is not something a port may read (as
  `make_write_population` already argues).
- **Field names.** The record's fields must not reuse any name a σ
  reader looks up through `optax.tree.get` (`count`, `mu`, `nu`, `t`,
  `step_sizes`, …). A duplicate name makes that lookup raise `KeyError`
  (checked against the installed optax).
- **Cost.** 2·10·d + 2·d numbers per optimizer, about 170 KB at
  d = 1024 in float64. Per step, one subtraction and two dot products.

### Ports per optimizer

Readers convert into the unit; writers convert out of it.

**Full-matrix slots**

| Optimizer | Reads | Writes |
|---|---|---|
| cmaes, iamalgamfull | `C`: diagonal, widest and narrowest | `C` |
| maes | `M`, as `MMᵀ` | `M` |
| xnes | `B`, as `BBᵀ` | `B` |
| bfgs, dfp | via the record | the starting inverse Hessian after a restart |

**Low-rank slots**

| Optimizer | Reads | Writes |
|---|---|---|
| crfmnes | `D`, `v` natively | `D`, plus `v` from the widest direction only |
| lmmaes | `M`'s directions | both directions |
| rmes | `P` | the widest direction only: its paths add variance, like `v` |

**Diagonal slots**

| Optimizer | Reads | Writes |
|---|---|---|
| sepcmaes, iamalgamunivariate | `C` | `C` |
| snes, simplees | `std`, squared | `std = √diag` |
| pgpe, des | none until their rates are pinned down | `std = √diag` |
| rprop | via the record | step-size ratios ∝ √diag, keeping its own RMS, clipped to its bounds |

**Gradient optimizers without a slot**

| Optimizer | Reads | Writes |
|---|---|---|
| lbfgs | its own pairs | nothing (see below) |
| the Adam family, sgd, lion and the other optax transforms | via the record | nothing |
| nonlinearcg, lbfgsb | via the record | nothing: they restart, with no slot |

RMSProp, Adagrad, Adafactor and SM3 also report no scale, so they
restart on every switch with news. For Adagrad that is a fix: its
`sum_of_squares` only grows, so after a far move it keeps shrinking its
steps based on gradients from somewhere else.

**Nelder–Mead**

Reads nothing: it has no rate, and a degenerate simplex is not a
reliable curvature estimate. It writes the fresh simplex after a
restart, oriented and stretched by the shape, at its own starting size.
This row lands with ADR 0011's Nelder–Mead stage. Until then
`neldermead` is plain-run (`optimizer_parts.py:106`).

**No shape**

These read nothing and write nothing. They still declare a scale
reader, a point reader and a restart.

- the ES with a single σ: openes, ars, esmc, gradientlessdescent,
  hillclimbing, simulatedannealing, noisereusees, persistentes;
- the GAs: mr15ga, samrga, gesmrga, simplega;
- DE, SHADE, PSO, TuRBO, the RBF trust region, randomsearch;
- **diffusionevolution:** its `latent_projection` is drawn once at
  `init` and never updated (`diffusion_evolution.py:111–115`);
- **asebo, guidedes:** their subspaces are built from *gradient*
  estimates (`asebo.py:100–125`, `guided_es.py:100–105`). That is
  first-order information, and it widens the search where the landscape
  is steep, which is the opposite of this unit.

**Why L-BFGS receives nothing.** optax's L-BFGS starts from γ·I. γ is
min(1, 1/|g|) on the first step, and from then on it is sᵀy/yᵀy of the
newest *real* pair, recomputed every step
(`optax/_src/transform.py:1770–1783`). A stored pair fixes an
*absolute* curvature along its step, so a seeded pair would need a
size, and a shape carries none. After one real step γ jumps to the
measured value, while the seeded pair keeps the old size, so for up to
9 more steps L-BFGS would treat that direction as far more or far less
curved than everything else. Its starting guess is a single number, so
the diagonal has nowhere to go either. BFGS and DFP do not have this
problem: their default starting guess is the identity, size 1, and a
det-1 shape only redistributes that.

**Why Adam receives nothing.** We measured that a write making Adam's
steps 1000× smaller held for all 200 steps tested. One making them
1000× larger gave 5× for one step and was gone within about 20. Fresh
gradients swamp a shrunken `nu`, while an inflated one decays only
slowly. A shape write would therefore leave Adam slower along the
directions marked curved, and do nothing else.

**Rprop receives in length units.** Its own ratchet overwrites the
write within a few steps, so the write is short-lived but harmless.

## Before building: the slope test

These checks must pass before the shape ports ship.

- **Slope.** On a separable ellipsoid with known curvatures h, plot
  log(reading) against log(h) for the diagonal. Every provider must give
  a slope of −1. That includes the record on Adam (lr 1e-2), on Rprop
  and on BFGS. A slope near 0 means "not yet learned", not "wrong
  unit". At 1 adaptation time the slope must already be close to −1, or
  l2co ADR 0022's evidence threshold moves.
- **Directions.** On an ellipsoid with one long and one short axis
  rotated off the coordinate axes, the widest and the narrowest
  directions must both be found.
- **Round trip.** For every receiver, read(write(shape)) returns the
  shape, for the parts that receiver can hold.
- **Coverage.** How many genuinely different directions the record's
  10 steps span. This is the record's main risk: Adam and Rprop take
  similarly-signed steps across all coordinates for long stretches, so
  their consecutive steps may be nearly parallel.
- **Rotated ellipsoid.** The pairs must not depend on the valley lining
  up with the coordinate axes, and the residual test must switch the
  fitted diagonal off when it does not fit. Here the record's diagonal
  on BFGS is also compared with BFGS's own inverse-Hessian diagonal. If
  the record's is clearly worse, the BFGS reader takes its diagonal
  from the inverse Hessian instead, labelled with the record's scale
  and point.
- **Noisy ellipsoid.** What fraction of pairs the filter rejects, and
  how far the slope degrades.
- **Identity.** Trajectories with and without the record are equal bit
  for bit, on the switching path.
- **Reader accuracy.** The 1.5× test above, for every gradient scale
  reader.

## Consequences

- **Amends l2co ADR 0013.** Its "if added, shape would be diagonal plus
  rank-1" becomes this ADR's layout. The σ readers stay, as measurement
  scale and as the distance gate's denominator.
- **`TransferSpec` grows** a restart declaration. `UpdateClass` and the
  `*Parts` gain a point reader, evidence rates and the shape ports.
- **The factory keyword convention grows** by one keyword,
  `curvature_record=`. Every factory takes it, and those without a
  gradient ignore it.
- **The Adam-family scale reader changes value** once it is
  bias-corrected. This only matters while the scale write still exists
  (`transfer_scale`, l2co ADR 0022). `headroom4` has no Adam member, so
  the scale A/B is not affected.
- **New bundle fields.** `TransferBundle` gains the shape (4d plus a
  few numbers), the measurement point (d) and the evidence per part.
- **Anything that walks an optax state by structure,** rather than
  through `optax.tree.get`, must look through the record's wrapper when
  the record is on.

## Considered options

- *Take shape only from optimizers that already hold one.* Rejected.
  It leaves Adam and Rprop silent, although they evaluate a gradient at
  every point they visit.
- *A widen-only layout with one direction (CR-FM-NES's exact form).*
  Superseded. It throws away the narrow direction for the six receivers
  that could use it.
- *A number of directions chosen per menu.* Rejected. It makes the
  bundle's layout depend on the menu, and two directions already cover
  the clear cases.
- *Gate each pair separately against the receiver's scale.* Rejected.
  The bundle would have to carry all 10 pairs instead of the compact
  shape.
- *Seed L-BFGS with pairs, and resize them once its first real pair
  arrives.* Rejected. It changes L-BFGS's own update, and keeping that
  update untouched is what lets the identity check pass.
- *For BFGS and DFP, have the switching layer keep a running step
  length and path midpoint since their last restart.* Rejected. BFGS
  keeps old directions indefinitely and replaces recent ones each step,
  so no single average describes where its shape was measured.
- *A generic fallback evidence rate such as 1/d.* Rejected. It is wrong
  by orders of magnitude for anything that is not CMA-like.
- *Record pairs across a stride of j steps.* Deferred. Decide from the
  coverage and noise numbers.
- *Re-evaluate the previous gradient with the same noise draw.*
  Rejected for now. It costs one extra evaluation per stored pair.
- *Enable the record only on noisy tasks, or only on noiseless ones.*
  Rejected: a real problem does not come labelled.
- *Make the record a hyperparameter.* Rejected. It would change the
  hash of every schedule that turns it on, although behaviour is
  identical.
- *Wrap every gradient transform, static runs included.* Rejected.
  Static runs never read it, and it would add memory to every databank
  run and a new way for stored rows to break.
- *Normalise by the trace instead of the geometric mean.* Rejected. A
  change of power breaks trace normalisation, so a unit conversion
  would also change the size.
