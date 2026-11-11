# Vendored reference: Zenodo `dynas` MLSL

The `mlsl` registry entry (`src/l2co_optimizers/_src/mlsl.py`) is a port
of the MLSL in this folder. `tests/test_mlsl.py` drives both with the same
random stream and checks that they evaluate the same points. This code is
a validation reference only. It is not shipped with the package.

## Source

- Record: Jankovic, A., Kostovska, A., Vermetten, D., de Nobel, J., Wang, H.,
  Eftimov, T. and Doerr, C. (2022). *Per-Run Algorithm Selection with
  Warm-starting using Trajectory-based Features - Data*. Zenodo.
  <https://doi.org/10.5281/zenodo.6458266>
- Archive: `data-collection.zip`, folder `dynas/`.
- Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  <https://creativecommons.org/licenses/by/4.0/>. The copies here keep the
  authors' code as published, apart from the edit listed below.

## Files

| File | Upstream | Edited |
| --- | --- | --- |
| `mlsl.py` | `dynas/mlsl.py` | no |
| `algorithm.py` | `dynas/algorithm.py` (base class `Algorithm`) | yes, see below |
| `parameters.py` | `dynas/parameters.py` (imported by `algorithm.py`) | no |
| `__init__.py` | new, only makes the folder a package | - |

SHA-256 of the upstream files, as copied:

```
798e7f3e9f04c751659423ce73b4bccabe6e490e4a8818eb00277646560c5f1f  mlsl.py
5c52fa696c080ac41e90a22b58e4a836ad33e2636c5585d4ebcaec8d6effdfa4  algorithm.py
8a2651e14f69401af360990b51826c794f5c51314ab44121d470f8060a265ef7  parameters.py
```

## Edits

1. `algorithm.py`, line 3: `from IOHexperimenter import IOH_function` is
   replaced by an empty stand-in class `IOH_function`. IOHexperimenter is not
   a dependency of this repo. The code only compares `type(func)` against
   this class, so with the stand-in every function counts as a "new IOH"
   function, which is the code path the test drives.

## How the test drives it

`tests/test_mlsl.py` gives the reference a small function object with the
attributes the new-IOH path reads (`meta_data.n_variables`,
`constraint.lb`/`ub`, `state.evaluations`). The reference draws its sample
points from the global `np.random` stream. The test seeds that stream
through the reference's own `seed=` argument, which accepts a
`np.random.get_state()` tuple: it passes the state of
`l2co_optimizers._src.mlsl.mlsl_rng(key)`, the generator the registry entry
draws from for the same JAX key.

The reference calls `scipy.optimize.minimize(method="Powell")`. The Zenodo
runs used scipy 1.7 or older, whose Powell checks `maxfev` only after a full
sweep, while the scipy this repo locks refuses `maxfev=0` outright. The test
therefore replaces the name `minimize` in the reference module (with
`monkeypatch`, so the file itself stays unedited) by
`l2co_optimizers._src.scipy17_powell.minimize_powell`, the copy of scipy
1.7.3's Powell that the registry entry uses.
