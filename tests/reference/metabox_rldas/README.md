# Vendored reference: MetaBox v2 RL-DAS differential-evolution variants

Parity reference for `jde21_metabox`, `madde_metabox` and
`nl_shade_rsp_metabox` (issues #60, #61, #62). This is a validation reference
only and is not shipped with the package.

## Source

- Repository: MetaEvo/MetaBox, branch `v2.0.0`, commit `5565a28`
  (<https://github.com/MetaEvo/MetaBox>).
- File: `src/environment/optimizer/rldas_optimizer.py`, SHA-256
  `9a69a9a8b4f189594f9d6ffb7e740d04b4fd8a06e0a469e19a34e08cb7b39646`.
- Licence: BSD 3-Clause, copied as `LICENSE`.

## What is copied, and the edits

`rldas_optimizer.py` holds upstream lines 1-3, 5-8 and 269-1502 unchanged:
the imports, `Population`, `NL_SHADE_RSP`, `JDE21`, `MadDE`, `Info` and the
landscape-feature helpers that `Population.get_feature` calls. The edits are
deletions only:

1. Line 4, `from .learnable_optimizer import Learnable_Optimizer`, removed
   (it pulls in the MetaBox package and torch).
2. Lines 9-268, class `RLDAS_Optimizer` (the PPO agent's environment
   wrapper), removed: it is not part of the DE variants and needs the
   import above.
3. Lines 1503-1551, the score-judge helpers, removed (unused here).

## Behaviour to know before using it

- Each DE's `step(population, problem, FEs, FEs_end, MaxFEs)` mutates the
  shared `Population` in place; the `Population` owns the shrink schedule
  (`NLPSR`, 170 -> 30 by default) for all three variants.
- Randomness comes from the `rng` passed to `Population` and the DEs
  (`np.random.RandomState`), except `NL_SHADE_RSP`'s Cauchy draws, which
  use the global `np.random` stream through `scipy.stats.cauchy.rvs`. Seed
  both for a reproducible reference run.
- The quirks a port must reproduce are listed in
  `~/wt/l2co-opt-wave-60-64/decisions.md`, "Reference facts", and must be
  repeated in each entry's module docstring.

## Fixtures

- `make_fixture_jde21.py` writes `tests/data/jde21_metabox_fixture.json`
  (MetaBox `JDE21`: final best cost of 20 seeds on sphere, Rastrigin and
  Rosenbrock, 10-D, box [-5, 5], 10 000 evaluations), read by the slow
  parity test in `tests/test_jde21.py`. Re-run it from the repository root
  with `uv run --no-sync python tests/reference/metabox_rldas/make_fixture_jde21.py`.
