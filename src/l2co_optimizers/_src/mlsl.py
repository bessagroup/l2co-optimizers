"""
MLSL -- Multi-Level Single Linkage, ported from the Zenodo ``dynas`` code.

``"mlsl"`` is the MLSL of Rinnooy Kan & Timmer (1987), "Stochastic global
optimization methods part II: Multi level methods" (Mathematical
Programming 39, 57-78, https://doi.org/10.1007/BF02592071), as implemented
in ``dynas/mlsl.py`` of Jankovic et al. (2022), *Per-Run Algorithm
Selection with Warm-starting using Trajectory-based Features - Data*,
Zenodo, https://doi.org/10.5281/zenodo.6458266 (``data-collection.zip``,
CC-BY-4.0). That file is the reference: this module ports it line for
line, and ``tests/test_mlsl.py`` checks, against a vendored copy
(``tests/reference/zenodo_mlsl``), that both evaluate the same points
when driven with the same random stream.

The algorithm. Each iteration ``k = 1, 2, ...``:

1. draw ``N = 50 d`` points uniformly in the box and evaluate them;
2. take the reduced sample: the best ``ceil(gamma k N)`` points of all
   points drawn so far;
3. compute the critical distance ``r_k``;
4. from every point of the reduced sample that has no better point of
   the reduced sample within ``r_k``, run a local search: scipy's Powell
   with the box, ``ftol = 1e-8`` and ``maxfev`` set to the local budget
   left.

Published parameters. The paper gives the reduced-sample fraction
``gamma`` and the critical distance
``r_k = pi^(-1/2) (Gamma(1 + d/2) m(S) sigma log(kN) / (kN))^(1/d)``,
with ``m(S)`` the Lebesgue measure (volume) of the feasible set and
``sigma > 0`` (written ``zeta`` here, as in the reference). It leaves
the values free.

Values the paper does not state, all taken from the Zenodo code:

* ``gamma = 0.1`` and ``zeta = 2.0`` (the hyperparameters' defaults);
* ``N = 50 d`` points per iteration (the reference credits "the original
  BBOB submission");
* the local solver, scipy's Powell, with ``ftol = 1e-8``;
* the local budget: one pool of ``0.1 (B - e)`` evaluations, with ``B``
  the budget and ``e`` the evaluations made before MLSL starts (zero in
  a standalone run), computed once. Each local search may use what is
  left of it (``maxfev``) and its evaluations are taken off it, floored
  at zero.

Deviations of the reference from the paper, reproduced here on purpose
(the port follows the reference, not the paper):

1. **Volume.** ``m(S)`` is the box diagonal, not its volume: the
   reference hard-codes ``sqrt(100 d)``, the diagonal of BBOB's
   ``[-5, 5]^d``. For a general box this port uses the analogue
   ``||ub - lb||_2``, which equals ``sqrt(100 d)`` on ``[-5, 5]^d``.
2. **Sample count.** ``kN`` in ``r_k`` is ``k`` times the number of
   points drawn so far, so ``k^2 N``, not ``kN``.
3. **Logarithm.** ``log1p(kN)`` (that is ``log(1 + kN)``), not
   ``log(kN)``.
4. **Start condition.** Only "no better reduced-sample point within
   ``r_k``" is checked. The paper also skips points a local search
   already started from and points near a local minimum found, so the
   reference searches again from a point that stays in the reduced
   sample in a later iteration.
5. **Budget.** The reference checks its stopping criterion only at the
   top of an iteration, so it can overrun its budget by up to an
   iteration. This entry runs on the scipy entries' driver (ADR 0002),
   which ends the run at exactly ``n_iterations`` evaluations, so the
   history is the reference's first ``n_iterations`` evaluations.
6. **A spent local-search pool ends the run.** Once the pool is zero,
   the next local search calls Powell with ``maxfev = 0``. The scipy
   this repo locks (1.18) refuses that before evaluating anything
   (``_MaxFuncCallError`` from its first call), and the reference has
   no handler, so the reference run ends there. This entry makes the
   same call, and the driver treats the error as any failed scipy run
   (ADR 0002): the rest of the budget re-evaluates the best point
   found. With the default ``gamma`` and ``zeta`` this happens within
   the first iteration on small test problems. Older scipy releases
   evaluated at least once before checking ``maxfev``; which release
   the Zenodo runs used is not recorded with the code.

How the l2co interface enters:

* **Every evaluation** -- each sample point, and each call Powell makes
  -- goes through the l2co loss and is one history entry, billed one
  evaluation (l2co ADR 0026), exactly as for the scipy entries. A run
  always lasts its whole budget: MLSL has no stopping rule of its own,
  and after deviation 6 the driver re-evaluates the best point.
* **Start point and scale are accepted and ignored.** MLSL has no warm
  start: the reference samples the whole box from its first iteration
  and ignores any state handed over at a switch (the CMA-ES hand-over
  of the Zenodo switching runs). The starting population is still
  required by the run loop, and it seeds only the driver's "best so far"
  until the first evaluation.
* **Randomness.** The reference draws from the global ``np.random``
  stream. This entry draws from its own :class:`numpy.random.RandomState`
  (:func:`mlsl_rng`), seeded from the run's key, and never touches the
  global stream. The same calls on the same state give the same draws,
  which is how the parity test drives the reference.
* **The box is required.** Sampling needs finite bounds on every side;
  a run without them raises before it starts. The factory itself accepts
  any ``bounded``, since callers such as ``resolve_popsize`` build every
  entry unbounded only to read its population size.
* **No stopping criterion and no switching**, as for the scipy entries:
  a ``stop_fn`` raises and ``optimizer_parts`` refuses ``"mlsl"``.

Non-goal. bessagroup/l2co-optimizers#63 proposed a JAX MLSL with
``bfgs`` as the local solver and a warm start from the CMA-ES state.
That is not what the Zenodo switching runs used, and is out of scope
here: this entry is the reference MLSL only.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
from typing import Any

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.optimize as so
from jax.tree_util import Partial
from jaxtyping import PyTree
from scipy.special import gamma as gamma_function

# Local
from l2co_optimizers._src.core.state_transfer import FAMILY_DERIVATIVE_FREE
from l2co_optimizers._src.core.typing import LossFunction, StopFunction
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.scipy_implementations import (
    ScipyUpdateClass,
    _Driver,
    _scipy_update,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Experimental"
# =============================================================================

#: The registry names this module provides, normalized.
MLSL_OPTIMIZERS: frozenset[str] = frozenset({"mlsl"})

#: Points drawn per iteration, per dimension (the Zenodo code).
_POINTS_PER_DIMENSION = 50

#: Share of the remaining budget that forms the local-search pool.
_LOCAL_BUDGET_SHARE = 0.1

#: Powell's ``ftol`` in a local search (the Zenodo code).
_LOCAL_FTOL = 1e-8

#: Folded into the run's key to seed the sampling generator, so its
#: stream is independent of the keys the loss evaluations split off.
_RNG_TAG = 0x4D4C534C  # "MLSL"


def mlsl_rng(key: Any) -> np.random.RandomState:
    """The generator ``"mlsl"`` draws its sample points from.

    A legacy :class:`numpy.random.RandomState`, as the reference draws
    from the legacy global stream, seeded with the key data of
    ``jax.random.fold_in(key, _RNG_TAG)``. Its
    :meth:`~numpy.random.RandomState.get_state` handed to
    :func:`numpy.random.set_state` makes the global stream draw the same
    numbers.

    Parameters
    ----------
    key
        The run's PRNG key, typed or raw.

    Returns
    -------
    numpy.random.RandomState
        A fresh generator; the same key always gives the same stream.
    """
    key = jr.fold_in(key, _RNG_TAG)
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jr.key_data(key)
    return np.random.RandomState(np.asarray(key, dtype=np.uint32).ravel())


def _critical_distance(
    k: int, n_points: int, dim: int, lebesgue: float, zeta: float
) -> float:
    """``r_k`` exactly as the reference computes it (deviations 1-3)."""
    kN = k * n_points
    return (1 / math.sqrt(np.pi)) * math.pow(
        (
            gamma_function(1 + (dim / 2))
            * lebesgue
            * (zeta * math.log1p(kN))
            / kN
        ),
        (1 / dim),
    )


def _mlsl_minimizer(gamma: float, zeta: float):
    """The ``minimize`` closure the driver runs: the reference's ``run``."""

    def minimize(driver: _Driver, x0: np.ndarray, bounds: so.Bounds):
        del x0  # no warm start: MLSL samples the whole box
        rng = mlsl_rng(driver.start_key)
        lb, ub = bounds.lb, bounds.ub
        dim = lb.size
        budget = driver.losses.size
        local_budget = _LOCAL_BUDGET_SHARE * (budget - driver.count)
        popsize = _POINTS_PER_DIMENSION * dim
        lebesgue = float(np.linalg.norm(ub - lb))
        pop: list[np.ndarray] = []
        f: list[float] = []
        k = 1
        # The driver ends the run (raising out of an evaluation) when the
        # budget is spent; MLSL has no stopping rule of its own. Powell
        # with a spent pool raises too (deviation 6).
        while True:
            for _ in range(popsize):
                new_point = rng.uniform(lb, ub)
                pop.append(new_point)
                f.append(driver.value(new_point))

            n_reduced = math.ceil(gamma * k * popsize)
            m = np.hstack((np.asarray(pop), np.expand_dims(np.asarray(f), 1)))
            sorted_m = m[np.argsort(m[:, dim])]
            xr = sorted_m[0:n_reduced, 0:dim]
            fr = sorted_m[0:n_reduced, dim]

            rk = _critical_distance(k, len(pop), dim, lebesgue, zeta)

            for i in range(len(xr)):
                cond = False
                for j in range(len(xr)):
                    if j == i:
                        continue
                    if fr[j] < fr[i]:
                        cond = np.linalg.norm(xr[j] - xr[i]) < rk
                    if cond:
                        break
                if not cond:
                    solution = so.minimize(
                        driver.value,
                        xr[i],
                        method="Powell",
                        bounds=bounds,
                        options={"ftol": _LOCAL_FTOL, "maxfev": local_budget},
                    )
                    local_budget = max(local_budget - solution.nfev, 0)
            k = k + 1

    return minimize


def mlsl_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    gamma: float = 0.1,
    zeta: float = 2.0,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="mlsl"``.

    Multi-Level Single Linkage as in the Zenodo ``dynas/mlsl.py``; see
    the module docstring for the algorithm, its sources and its
    deviations from the paper. It runs its whole budget in one host
    callback, on the scipy entries' driver (ADR 0002). The starting
    point is accepted and ignored.

    Parameters
    ----------
    model : PyTree
        Model whose inexact-array leaves are optimized; the rest is
        recombined as static structure.
    loss_fn : LossFunction
        ``loss_fn(model, **sample)`` -- or ``loss_fn(model, key=key,
        **sample)`` when ``pass_rng`` -- returning a scalar loss.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword (a stochastic loss).
    opt_hash : int
        Stable hash stamped into ``HistoryState.update_step``.
    bounded : tuple[float, float]
        The box ``(lower, upper)``, applied to every coordinate. Both
        sides must be finite: MLSL samples it uniformly.
    stop_fn : StopFunction or None, optional
        Must be ``None``; see Raises.
    gamma : float, optional
        Fraction of the points drawn so far that forms the reduced
        sample; ``0.1`` (the Zenodo code).
    zeta : float, optional
        Scaling factor of the critical distance (the paper's ``sigma``);
        ``2.0`` (the Zenodo code).

    Returns
    -------
    UpdateClass
        A ``ScipyUpdateClass`` (ADR 0002).

    Raises
    ------
    ValueError
        If ``stop_fn`` is not ``None``; from ``run``, if ``bounded`` is
        not finite on both sides.
    """
    scipy_update = _scipy_update(
        _mlsl_minimizer(gamma, zeta),
        name="mlsl",
        family=FAMILY_DERIVATIVE_FREE,
        with_grad=False,
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )
    return MLSLUpdateClass(
        driver=scipy_update.driver, hash=opt_hash, bounded=bounded
    )


class MLSLUpdateClass(ScipyUpdateClass):
    """A :class:`ScipyUpdateClass` that refuses to run without a finite box.

    The check runs in :meth:`run`, at trace time and before the host
    callback, where the driver would otherwise swallow it as a failed run.
    """

    bounded: tuple[float | None, float | None] | None = eqx.field(static=True)

    def __init__(self, driver, hash: int, bounded):
        super().__init__(
            driver=driver, hash=hash, family=FAMILY_DERIVATIVE_FREE
        )
        self.bounded = bounded

    def run(self, *args, **kwargs):
        """:meth:`ScipyUpdateClass.run`, after checking the box.

        Raises
        ------
        ValueError
            If ``bounded`` is not finite on both sides.
        """
        lo, hi = (None, None) if self.bounded is None else self.bounded
        if lo is None or hi is None or not np.isfinite([lo, hi]).all():
            raise ValueError(
                f"mlsl samples the box uniformly, so it needs finite "
                f"bounds on both sides; got bounded={self.bounded!r}."
            )
        return super().run(*args, **kwargs)


mlsl_mapping = {"mlsl": Partial(mlsl_update)}
