"""
scipy minimisers stepped one evaluation at a time (ADR 0011).

scipy's L-BFGS-B Fortran routine (``scipy.optimize._lbfgsb.setulb``) and
its SLSQP routine (``scipy.optimize._slsqplib.slsqp``) use *reverse
communication*: they return to their caller every time they need the
objective, with all of their state in the arrays (and, for SLSQP, the
dict) they were handed. This module drives them from inside a JAX loop,
one evaluation per step, so ``"lbfgsb"`` and ``"slsqp"`` can sit in a
switching menu. They replace ADR 0002's whole-run driver
(:mod:`~l2co_optimizers._src.scipy_implementations`) under the same
names and hashes.

One transform backs both construction paths:

* :func:`lbfgsb_transform` and :func:`slsqp_transform` are
  :class:`optax.GradientTransformationExtraArgs` whose ``params`` is
  always the point scipy evaluates next -- the rule ADR 0011 sets for
  every solver that evaluates the objective itself. ``update`` takes the
  loss (``value=``) and gradient there, hands them to scipy, and runs
  scipy until it asks for the objective at a new point.
* :func:`lbfgsb_update` and :func:`slsqp_update`, the registry
  factories, wrap that transform in an ordinary
  :class:`~l2co_optimizers._src.core.update_class.UpdateClass` whose step
  evaluates the loss in the scan, as every other gradient entry does.
  :func:`~l2co_optimizers._src.optimizer_parts.optimizer_parts` hands the
  same transform to a switching loop as ``GradientParts``.

The decisions, in brief:

* **scipy's state rides in the carry as opaque bits.** Every array the
  routine keeps -- the float64 iterate and workspaces, the integer save
  arrays, SLSQP's state dict -- is viewed as ``uint32`` on the host and
  concatenated (:class:`_Layout`), so the carry holds float64 state
  whether or not JAX runs in x64. The point the loop evaluates is the
  float64 iterate cast to the loop's dtype, as ADR 0002's
  ``jnp.asarray(x)`` cast it.
* **Each step is one host callback** (:func:`jax.pure_callback`) that
  feeds scipy the loss and gradient and runs it to its next request. A
  request at the point just evaluated -- L-BFGS-B's re-request after
  ``START``, SLSQP's separate gradient request -- is answered from those
  values without a new step, as scipy's own ``ScalarFunction`` cache
  answered it for ADR 0002's driver. Under ``vmap`` the callback runs
  once per step for the whole batch (``vmap_method="expand_dims"``),
  looping over it on the host.
* **ADR 0002's run rules carry over.** A run lasts its whole budget:
  once scipy stops, every step re-evaluates its final point, billed. A
  failure -- scipy raising, or proposing or returning a non-finite
  point -- ends the run at the best point evaluated so far, re-evaluated
  to the budget. A NaN loss is passed to scipy unchanged and never
  becomes the best.
* **Bounds** reach scipy natively on the registry path, so every
  iterate stays in the box. The parts path has no bounds channel yet
  (ADR 0011, "Bounds are deferred"): a switched entry is unbounded.
* ``_lbfgsb.setulb`` and ``_slsqplib.slsqp`` are private scipy API. The
  tests pin their behaviour against ADR 0002's driver bitwise.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, NamedTuple

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, UInt32
from scipy.linalg.lapack import HAS_ILP64
from scipy.optimize import _lbfgsb, _slsqplib

# Local
from l2co_optimizers._src.core.opt_history import OptHistory
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_GRADIENT,
    build_transfer_fns,
)
from l2co_optimizers._src.core.typing import (
    InputParameters,
    LossFunction,
    StopFunction,
)
from l2co_optimizers._src.core.update_class import UpdateClass
from l2co_optimizers._src.scipy_implementations import (
    _LBFGSB_FTOL,
    _LBFGSB_GTOL,
    _SLSQP_FTOL,
    _UNREACHABLE,
    _loss_at,
    _np_bounds,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

logger = logging.getLogger(__name__)

#: The registry names this module provides, normalized.
STEPPED_SCIPY_OPTIMIZERS: frozenset[str] = frozenset({"lbfgsb", "slsqp"})

#: scipy's integer dtype for the Fortran workspaces.
_INT = np.int64 if HAS_ILP64 else np.int32

#: ``status`` values: scipy is still asking for points, or it has
#: stopped (or failed) and the run re-evaluates ``req`` to the budget.
_RUNNING = 0
_IDLE = 1

#: ``setulb``'s ``task[0]`` codes (``scipy.optimize._lbfgsb_py``'s
#: ``status_messages``).
_TASK_FG = 3
_TASK_NEW_X = 1

#: SLSQP's ``mode`` requests: the objective, or its gradient, at ``x``.
_MODE_F = 1
_MODE_G = -1

#: What the next evaluation is for (SLSQP's ``pending``): the start, an
#: objective request or a gradient request.
_FOR_START = 0
_FOR_F = 1
_FOR_G = 2

#: The keys of SLSQP's state dict -- the C struct ``SLSQP_static_vars``
#: plus ``m``, ``meq`` and ``n`` -- split by type, in packing order.
_SLSQP_FLOATS = (
    "acc", "alpha", "f0", "gs", "h1", "h2", "h3", "h4", "t", "t0", "tol",
)  # fmt: skip
_SLSQP_INTS = (
    "exact", "inconsistent", "reset", "iter", "itermax", "line", "m",
    "meq", "mode", "n",
)  # fmt: skip


#                                                          Packed host state
# =============================================================================


class _Layout:
    """How a dict of host arrays packs into one ``uint32`` vector.

    Each field is a numpy array of a fixed dtype and length; packing
    views its bytes as ``uint32`` and concatenates, so the vector can
    hold float64 and int64 state in a JAX carry without x64. Every
    field's byte size is a multiple of four.

    Parameters
    ----------
    fields : tuple[tuple[str, numpy.dtype, int], ...]
        ``(name, dtype, length)`` per field, in packing order.
    """

    def __init__(self, fields: tuple[tuple[str, Any, int], ...]):
        self.fields = tuple(
            (name, np.dtype(dtype), int(length))
            for name, dtype, length in fields
        )
        words = [d.itemsize * length // 4 for _, d, length in self.fields]
        self.offsets = np.concatenate([[0], np.cumsum(words)]).astype(int)
        self.size = int(self.offsets[-1])

    def pack(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        """``arrays`` as one ``uint32`` vector of length :attr:`size`."""
        return np.concatenate(
            [
                np.ascontiguousarray(arrays[name], dtype=dtype)
                .reshape(length)
                .view(np.uint32)
                for name, dtype, length in self.fields
            ]
        )

    def unpack(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """The fields of ``packed``, as writable copies."""
        packed = np.ascontiguousarray(packed, dtype=np.uint32)
        return {
            name: packed[start:stop].view(dtype).copy()
            for (name, dtype, _), start, stop in zip(
                self.fields, self.offsets[:-1], self.offsets[1:], strict=True
            )
        }


def _common_fields(n: int) -> tuple[tuple[str, Any, int], ...]:
    """The fields every stepped method keeps, for ``n`` dimensions.

    ``x`` is scipy's own iterate; ``req`` the point the loop evaluates
    next (``x`` while scipy runs, the final or best point once it idles);
    ``last``, ``f_last`` and ``g_last`` the previous evaluation, which
    answers a request at that same point; ``best_x`` and ``best_loss``
    the lowest loss so far.
    """
    f64 = np.float64
    return (
        ("x", f64, n),
        ("req", f64, n),
        ("last", f64, n),
        ("f_last", f64, 1),
        ("g_last", f64, n),
        ("best_x", f64, n),
        ("best_loss", f64, 1),
    )


def _start(layout: _Layout, x: np.ndarray) -> dict[str, np.ndarray]:
    """A fresh host state at ``x``: nothing evaluated, scipy running."""
    n = x.size
    state = {name: np.zeros(length, d) for name, d, length in layout.fields}
    state.update(
        x=x,
        req=x,
        best_x=x,
        best_loss=np.array([np.inf]),
        f_last=np.array([np.nan]),
        g_last=np.full(n, np.nan),
        last=np.full(n, np.nan),
        status=np.array([_RUNNING]),
    )
    return state


def _record(
    s: dict[str, np.ndarray],
    f: float,
    g: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> None:
    """Note the evaluation at ``req`` as the last one, and maybe the best."""
    s["last"] = s["req"].copy()
    s["f_last"][0] = f
    s["g_last"] = g.copy()
    # A NaN loss never wins; ties keep the earliest point.
    if f < s["best_loss"][0]:
        s["best_loss"][0] = f
        s["best_x"] = np.clip(s["req"], lo, hi)


def _stopped(
    s: dict[str, np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, int]:
    """scipy stopped at ``x``: idle there, or at the best if not finite."""
    if np.all(np.isfinite(x)):
        return x.copy(), _IDLE
    return _failed(s, "scipy returned a non-finite x")


def _failed(s: dict[str, np.ndarray], reason: str) -> tuple[np.ndarray, int]:
    """End the run at the best point evaluated so far."""
    logger.debug("scipy failed: %s", reason)
    return s["best_x"].copy(), _IDLE


@functools.cache
def _lbfgsb_layout(n: int, m: int) -> _Layout:
    """The packed state of an ``n``-dimensional L-BFGS-B with memory ``m``.

    The common fields (:func:`_common_fields`), then ``setulb``'s
    workspaces, sized as ``scipy.optimize._lbfgsb_py._minimize_lbfgsb``
    sizes them.
    """
    f64 = np.float64
    return _Layout(
        (
            *_common_fields(n),
            ("wa", f64, 2 * m * n + 5 * n + 11 * m * m + 8 * m),
            ("dsave", f64, 29),
            ("iwa", _INT, 3 * n),
            ("task", _INT, 2),
            ("ln_task", _INT, 2),
            ("lsave", _INT, 4),
            ("isave", _INT, 44),
            ("status", _INT, 1),
        )
    )


@functools.cache
def _lbfgsb_bounds(
    bounded: tuple[float | None, float | None], n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(lo, hi, l, u, nbd)``: the clip box and ``setulb``'s bound arrays.

    ``nbd`` codes each coordinate as ``_minimize_lbfgsb`` does: 0
    unbounded, 1 lower only, 2 both, 3 upper only; ``l`` and ``u`` are
    zero where a side is unbounded.
    """
    lo, hi = _np_bounds(bounded, n)
    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    nbd = np.select(
        [has_lo & has_hi, has_lo, has_hi], [2, 1, 3], default=0
    ).astype(_INT)
    lower = np.where(has_lo, lo, 0.0)
    upper = np.where(has_hi, hi, 0.0)
    return lo, hi, lower, upper, nbd


#                                                      Host-side L-BFGS-B
# =============================================================================


def _lbfgsb_host_init(
    x0: np.ndarray, *, m: int, bounded: tuple[float | None, float | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Start one realization: ``(req, packed)``.

    The starting point is clipped into the box, as ``minimize`` clips
    it; it is the first point evaluated, as ``ScalarFunction`` evaluates
    it before ``setulb`` is first called.
    """
    lo, hi, *_ = _lbfgsb_bounds(bounded, x0.size)
    layout = _lbfgsb_layout(x0.size, m)
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    return x, layout.pack(_start(layout, x))


def _lbfgsb_host_update(
    packed: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    *,
    m: int,
    maxls: int,
    bounded: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Hand scipy the evaluation at ``req``: ``(next req, packed)``.

    Runs ``setulb`` until it asks for the objective at a point other
    than the one just evaluated, or stops. Never raises: any failure
    ends the run at the best point.
    """
    n = g.size
    lo, hi, lower, upper, nbd = _lbfgsb_bounds(bounded, n)
    layout = _lbfgsb_layout(n, m)
    s = layout.unpack(packed)

    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    _record(s, f, g, lo, hi)
    if s["status"][0] == _RUNNING:
        s["req"], s["status"][0] = _lbfgsb_advance(
            s, f, g, m=m, maxls=maxls, lower=lower, upper=upper, nbd=nbd
        )
    return s["req"], layout.pack(s)


def _lbfgsb_advance(
    s: dict[str, np.ndarray],
    f: float,
    g: np.ndarray,
    *,
    m: int,
    maxls: int,
    lower: np.ndarray,
    upper: np.ndarray,
    nbd: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Run ``setulb`` to its next request: ``(point, status)``."""
    x = s["x"]
    factr = _LBFGSB_FTOL / np.finfo(float).eps
    try:
        while True:
            _lbfgsb.setulb(
                m,
                x,
                lower,
                upper,
                nbd,
                f,
                g,
                factr,
                _LBFGSB_GTOL,
                s["wa"],
                s["iwa"],
                s["task"],
                s["lsave"],
                s["isave"],
                s["dsave"],
                maxls,
                s["ln_task"],
            )
            task = int(s["task"][0])
            if task == _TASK_NEW_X:
                continue
            if task != _TASK_FG:
                logger.debug("scipy stopped: task %s", s["task"])
                return _stopped(s, x)
            if not np.all(np.isfinite(x)):
                return _failed(s, "scipy proposed a non-finite x")
            if not np.array_equal(x, s["last"]):
                return x.copy(), _RUNNING
            # scipy's ScalarFunction answers this from its cache.
            f, g = float(s["f_last"][0]), s["g_last"].copy()
    except Exception as error:  # noqa: BLE001 -- any scipy failure ends it
        return _failed(s, f"scipy raised {error!r}")


#                                                         Host-side SLSQP
# =============================================================================


def _slsqp_buffer_size(n: int) -> int:
    """SLSQP's float workspace with no constraints but the box.

    ``_minimize_slsqp``'s formula at ``m = meq = 0``, topped up for the
    missing inequality constraints as it tops it up.
    """
    return n * (n + 1) // 2 + 8 * n * n + 35 * n + 28 + 2 * n * (n + 1)


@functools.cache
def _slsqp_layout(n: int) -> _Layout:
    """The packed state of an ``n``-dimensional SLSQP.

    The common fields (:func:`_common_fields`); ``fx`` and ``g``, the
    objective and gradient scipy is handed next (a gradient request
    leaves ``fx`` as it was, as scipy's own loop does); the state dict
    (``floats``, ``ints``); the multipliers and workspaces, sized as
    ``scipy.optimize._slsqp_py._minimize_slsqp`` sizes them; and
    ``pending``, what the next evaluation is for.
    """
    f64, i64 = np.float64, np.int64
    return _Layout(
        (
            *_common_fields(n),
            ("fx", f64, 1),
            ("g", f64, n),
            ("floats", f64, len(_SLSQP_FLOATS)),
            ("mult", f64, 2 * n + 2),
            ("buffer", f64, _slsqp_buffer_size(n)),
            ("ints", i64, len(_SLSQP_INTS)),
            ("indices", _INT, 2 * n + 2),
            ("status", _INT, 1),
            ("pending", _INT, 1),
        )
    )


@functools.cache
def _slsqp_bounds(
    bounded: tuple[float | None, float | None], n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(lo, hi, xl, xu)``: the clip box and SLSQP's bound arrays.

    SLSQP marks an unbounded side with NaN, as ``_minimize_slsqp`` does.
    """
    lo, hi = _np_bounds(bounded, n)
    xl = np.where(np.isfinite(lo), lo, np.nan)
    xu = np.where(np.isfinite(hi), hi, np.nan)
    return lo, hi, xl, xu


def _slsqp_host_init(
    x0: np.ndarray, *, bounded: tuple[float | None, float | None]
) -> tuple[np.ndarray, np.ndarray]:
    """Start one realization: ``(req, packed)``.

    As for L-BFGS-B, the starting point is clipped into the box and is
    the first point evaluated. The state dict starts as
    ``_minimize_slsqp`` starts it, with zero tolerances and an
    unreachable iteration cap.
    """
    n = x0.size
    lo, hi, *_ = _slsqp_bounds(bounded, n)
    layout = _slsqp_layout(n)
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    state = _start(layout, x)
    floats = dict.fromkeys(_SLSQP_FLOATS, 0.0)
    floats.update(acc=_SLSQP_FTOL, tol=10.0 * _SLSQP_FTOL)
    ints = dict.fromkeys(_SLSQP_INTS, 0)
    ints.update(itermax=_UNREACHABLE, n=n)
    state.update(
        floats=np.array([floats[k] for k in _SLSQP_FLOATS]),
        ints=np.array([ints[k] for k in _SLSQP_INTS]),
        pending=np.array([_FOR_START]),
    )
    return x, layout.pack(state)


def _slsqp_host_update(
    packed: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    *,
    bounded: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray]:
    """Hand scipy the evaluation at ``req``: ``(next req, packed)``.

    Runs ``slsqp`` until it asks for the objective or gradient at a
    point other than the one just evaluated, or stops. Never raises:
    any failure ends the run at the best point.
    """
    n = g.size
    lo, hi, xl, xu = _slsqp_bounds(bounded, n)
    layout = _slsqp_layout(n)
    s = layout.unpack(packed)

    f = float(f)
    g = np.asarray(g, dtype=np.float64)
    _record(s, f, g, lo, hi)
    if s["status"][0] == _RUNNING:
        pending = s["pending"][0]
        if pending in (_FOR_START, _FOR_F):
            s["fx"][0] = f
        if pending in (_FOR_START, _FOR_G):
            s["g"] = g.copy()
        s["req"], s["status"][0] = _slsqp_advance(s, xl=xl, xu=xu)
    return s["req"], layout.pack(s)


def _slsqp_advance(
    s: dict[str, np.ndarray], *, xl: np.ndarray, xu: np.ndarray
) -> tuple[np.ndarray, int]:
    """Run ``slsqp`` to its next request: ``(point, status)``."""
    n = s["x"].size
    x = s["x"]
    state = {
        k: float(v) for k, v in zip(_SLSQP_FLOATS, s["floats"], strict=True)
    }
    state |= {k: int(v) for k, v in zip(_SLSQP_INTS, s["ints"], strict=True)}
    # No constraints but the box: scipy hands it empty ones.
    constraints = np.zeros((1, n), order="F")
    values = np.zeros(1)
    try:
        while True:
            _slsqplib.slsqp(
                state,
                float(s["fx"][0]),
                s["g"],
                constraints,
                values,
                x,
                s["mult"],
                xl,
                xu,
                s["buffer"],
                s["indices"],
            )
            mode = state["mode"]
            if mode not in (_MODE_F, _MODE_G):
                logger.debug("scipy stopped: mode %s", mode)
                result = _stopped(s, x)
                break
            if not np.all(np.isfinite(x)):
                result = _failed(s, "scipy proposed a non-finite x")
                break
            if not np.array_equal(x, s["last"]):
                s["pending"][0] = _FOR_F if mode == _MODE_F else _FOR_G
                result = x.copy(), _RUNNING
                break
            # scipy's ScalarFunction answers this from its cache.
            if mode == _MODE_F:
                s["fx"][0] = s["f_last"][0]
            else:
                s["g"] = s["g_last"].copy()
    except Exception as error:  # noqa: BLE001 -- any scipy failure ends it
        result = _failed(s, f"scipy raised {error!r}")
    s["floats"] = np.array([state[k] for k in _SLSQP_FLOATS])
    s["ints"] = np.array([state[k] for k in _SLSQP_INTS])
    return result


#                                                    Callbacks under vmap
# =============================================================================


def _host_map(
    fn: Callable[..., tuple[np.ndarray, ...]],
    core_ndims: tuple[int, ...],
    out_shapes: tuple[jax.ShapeDtypeStruct, ...],
) -> Callable[..., tuple[np.ndarray, ...]]:
    """Run ``fn`` once per batch element of a ``vmap``-ed callback.

    Under ``vmap_method="expand_dims"`` an argument arrives with one
    leading axis per enclosing ``vmap`` -- the batch size where it is
    batched, one where it is not. Each argument's trailing
    ``core_ndims`` axes are its own; the leading ones broadcast to the
    batch shape, and ``fn`` runs on every element.
    """

    # Resolved now, at trace time: the callback may run where a scoped
    # ``jax.enable_x64`` no longer applies.
    specs = [(tuple(s.shape), np.dtype(s.dtype)) for s in out_shapes]

    def host(*args):
        args = [np.asarray(a) for a in args]
        leads = [
            a.shape[: a.ndim - k]
            for a, k in zip(args, core_ndims, strict=True)
        ]
        batch = np.broadcast_shapes(*leads)
        outs = [np.empty(batch + shape, dtype) for shape, dtype in specs]
        for idx in np.ndindex(*batch):
            element = [
                np.broadcast_to(a, batch + a.shape[a.ndim - k :])[idx]
                for a, k in zip(args, core_ndims, strict=True)
            ]
            for out, value in zip(outs, fn(*element), strict=True):
                out[idx] = value
        return tuple(outs)

    return host


#                                                               Transform
# =============================================================================


class SteppedScipyState(eqx.Module):
    """Optimizer state of a stepped scipy minimiser.

    Attributes
    ----------
    x : Float[Array, " n"]
        The point scipy evaluates next, flattened, in the loop's dtype.
    x_prev : Float[Array, " n"]
        The point evaluated before it; equal to ``x`` before the first
        step and while the run idles.
    packed : UInt32[Array, " k"]
        scipy's own state, as opaque bits (:class:`_Layout`).
    """

    x: Float[Array, " n"]
    x_prev: Float[Array, " n"]
    packed: UInt32[Array, " k"]


def _words_shape(n: int, dtype: np.dtype) -> jax.ShapeDtypeStruct:
    """The ``uint32`` words that carry ``n`` values of ``dtype``."""
    if dtype.itemsize % 4:
        raise TypeError(
            f"a stepped scipy minimiser needs a 32- or 64-bit float "
            f"model; got {dtype}"
        )
    return jax.ShapeDtypeStruct((n * dtype.itemsize // 4,), jnp.uint32)


def _to_words(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """``x`` cast to ``dtype``, as ``uint32`` words (host side)."""
    return np.ascontiguousarray(x, dtype=dtype).view(np.uint32)


def _from_words(words: UInt32[Array, "... w"], dtype: np.dtype) -> Array:
    """The values :func:`_to_words` packed, back in ``dtype``.

    A callback hands back ``uint32`` only: JAX checks a callback's
    float64 output against the x64 setting of the thread it runs in,
    which a scoped ``jax.enable_x64`` does not reach.
    """
    per = dtype.itemsize // 4
    if per > 1:
        words = words.reshape(*words.shape[:-1], -1, per)
    return jax.lax.bitcast_convert_type(words, dtype)


class _HostFns(NamedTuple):
    """The host side of one method: its init, update and packed size."""

    init: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    update: Callable[..., tuple[np.ndarray, np.ndarray]]
    packed_size: Callable[[int], int]


def _stepped_transform(
    host: _HostFns,
) -> optax.GradientTransformationExtraArgs:
    """An optax transform that steps a host-side scipy routine.

    ``init(params)`` starts scipy at ``params``; ``update(grads, state,
    params, *, value)`` hands it the loss and gradient at ``state.x`` and
    returns ``state.x_next - params``.
    """

    def init_fn(params: PyTree) -> SteppedScipyState:
        flat, _ = ravel_pytree(params)
        n = flat.shape[0]
        dtype = np.dtype(flat.dtype)
        shapes = (
            _words_shape(n, dtype),
            jax.ShapeDtypeStruct((host.packed_size(n),), jnp.uint32),
        )

        def call(x0):
            x, packed = host.init(x0)
            return _to_words(x, dtype), packed

        words, packed = jax.pure_callback(
            _host_map(call, (1,), shapes),
            shapes,
            flat,
            vmap_method="expand_dims",
        )
        x = _from_words(words, dtype)
        return SteppedScipyState(x=x, x_prev=x, packed=packed)

    def update_fn(
        updates: PyTree,
        state: SteppedScipyState,
        params: PyTree | None = None,
        *,
        value: Float[Array, ""] | None = None,
        **extra_args: Any,
    ) -> tuple[PyTree, SteppedScipyState]:
        del extra_args
        if value is None or params is None:
            raise ValueError(
                "A scipy minimiser unpacked into parts needs the loss at "
                "params: call update(grads, state, params, value=loss) "
                "(ADR 0011). A consumer that does not forward value= "
                "cannot host it."
            )
        grad, _ = ravel_pytree(updates)
        dtype = np.dtype(state.x.dtype)
        shapes = (
            _words_shape(state.x.shape[0], dtype),
            jax.ShapeDtypeStruct(state.packed.shape, jnp.uint32),
        )

        def call(packed, f, g):
            x, packed = host.update(packed, f, g)
            return _to_words(x, dtype), packed

        words, packed = jax.pure_callback(
            _host_map(call, (1, 0, 1), shapes),
            shapes,
            state.packed,
            jnp.asarray(value),
            grad,
            vmap_method="expand_dims",
        )
        x = _from_words(words, dtype)
        flat_params, unravel = ravel_pytree(params)
        step = unravel(x.astype(flat_params.dtype) - flat_params)
        return step, SteppedScipyState(x=x, x_prev=state.x, packed=packed)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def lbfgsb_transform(
    maxcor: int = 10,
    maxls: int = 20,
    bounded: tuple[float | None, float | None] = (None, None),
) -> optax.GradientTransformationExtraArgs:
    """scipy's L-BFGS-B as a stepped optax transform.

    Parameters
    ----------
    maxcor : int, optional
        Correction pairs kept by the limited-memory Hessian; scipy's
        default ``10``.
    maxls : int, optional
        Line-search steps per iteration; scipy's default ``20``.
    bounded : tuple[float | None, float | None], optional
        Box handed to scipy. The parts path leaves it unbounded.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        ``init(params)`` and ``update(grads, state, params, *, value)``,
        where ``grads`` and ``value`` are the gradient and loss at
        ``state.x``. ``update`` raises a ``ValueError`` without
        ``value`` or ``params``.

    Raises
    ------
    ValueError
        If ``maxcor`` or ``maxls`` is below one. scipy would raise only
        once the run had started, where it would look like a failed run.
    """
    if maxcor < 1 or maxls < 1:
        raise ValueError(
            f"lbfgsb needs maxcor >= 1 and maxls >= 1, got maxcor={maxcor} "
            f"and maxls={maxls}."
        )
    bounded = (None, None) if bounded is None else tuple(bounded)
    return _stepped_transform(
        _HostFns(
            init=functools.partial(
                _lbfgsb_host_init, m=maxcor, bounded=bounded
            ),
            update=functools.partial(
                _lbfgsb_host_update, m=maxcor, maxls=maxls, bounded=bounded
            ),
            packed_size=lambda n: _lbfgsb_layout(n, maxcor).size,
        )
    )


def slsqp_transform(
    bounded: tuple[float | None, float | None] = (None, None),
) -> optax.GradientTransformationExtraArgs:
    """scipy's SLSQP as a stepped optax transform.

    Parameters
    ----------
    bounded : tuple[float | None, float | None], optional
        Box handed to scipy. The parts path leaves it unbounded.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        ``init(params)`` and ``update(grads, state, params, *, value)``,
        where ``grads`` and ``value`` are the gradient and loss at
        ``state.x``. ``update`` raises a ``ValueError`` without
        ``value`` or ``params``.
    """
    bounded = (None, None) if bounded is None else tuple(bounded)
    return _stepped_transform(
        _HostFns(
            init=functools.partial(_slsqp_host_init, bounded=bounded),
            update=functools.partial(_slsqp_host_update, bounded=bounded),
            packed_size=lambda n: _slsqp_layout(n).size,
        )
    )


#                                                                 Factories
# =============================================================================


def _stepped_update(
    name: str,
    transform: optax.GradientTransformationExtraArgs,
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None],
    stop_fn: StopFunction | None,
) -> UpdateClass:
    """Wrap a stepped scipy ``transform`` into an ``UpdateClass``.

    Each step evaluates the loss and gradient at ``opt_state.x`` exactly
    as ADR 0002's driver did -- the point projected into ``bounded``,
    the gradient through the projection, the batch drawn by the run loop
    and the key split as ``new_key, eval_key`` -- and hands them to
    ``transform.update``. One step is one evaluation, billed 1.
    """
    read_fn, write_fn = build_transfer_fns(name, FAMILY_GRADIENT)
    model_params, _ = eqx.partition(model, eqx.is_inexact_array)
    _, unravel = ravel_pytree(model_params)
    loss_at = _loss_at(model, loss_fn, pass_rng, bounded)

    def expand(tree: PyTree) -> PyTree:
        return jax.tree.map(lambda a: jnp.expand_dims(a, 0), tree)

    def init_fn(
        params: InputParameters, key: PRNGKeyArray | None = None, **sample
    ) -> SteppedScipyState:
        flat, _ = ravel_pytree(jax.tree.map(lambda p: p[0], params))
        return transform.init(flat)

    def step_fn(
        carry: tuple[InputParameters, SteppedScipyState, PRNGKeyArray],
        sample: dict,
    ) -> tuple[
        tuple[InputParameters, SteppedScipyState, PRNGKeyArray], OptHistory
    ]:
        _, opt_state, key = carry
        new_key, eval_key = jr.split(key)
        x = opt_state.x
        loss, grad = jax.value_and_grad(loss_at)(x, sample, eval_key)
        _, opt_state = transform.update(grad, opt_state, x, value=loss)
        history = OptHistory(
            loss=jnp.expand_dims(loss, 0),
            params=expand(unravel(jnp.clip(x, *bounded))),
            grads=expand(unravel(grad)),
            update_step=jnp.array(opt_hash, dtype=int),
            fevals=jnp.array(1, dtype=int),
            iterations=jnp.array(1, dtype=int),
        )
        new_params = expand(unravel(opt_state.x))
        return (new_params, opt_state, new_key), history

    return UpdateClass(
        init_fn=init_fn,
        step_fn=step_fn,
        popsize=1,
        hash=opt_hash,
        stop_fn=stop_fn,
        family=FAMILY_GRADIENT,
        transfer_read_fn=read_fn,
        transfer_write_fn=write_fn,
    )


def lbfgsb_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
    maxcor: int = 10,
    maxls: int = 20,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="lbfgsb"``.

    scipy's L-BFGS-B (``minimize(method="L-BFGS-B")``): the
    limited-memory BFGS code of Byrd, Lu, Nocedal and Zhu, with a
    Moré--Thuente line search. It handles a box inside the method: a
    generalized Cauchy point along the projected gradient, then a
    minimisation over the variables not held at a bound, with a
    limited-memory model that knows which bounds are active. ``"lbfgs"``
    (optax's L-BFGS) instead clips into the box after each step.

    scipy is stepped one evaluation at a time (:func:`lbfgsb_transform`,
    ADR 0011), so the entry can sit in a switching menu and honours a
    ``stop_fn``. Without one, a run lasts its whole budget: once scipy
    stops, every remaining step re-evaluates its final point. Its
    stopping tolerances are zero (ADR 0006), so it stops only on a zero
    projected gradient, a zero relative reduction or a failed line
    search.

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
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box handed to scipy, which keeps every iterate inside it; the
        starting point is clipped into it. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function; ``None`` never stops.
    maxcor : int, optional
        Correction pairs kept by the limited-memory Hessian; scipy's
        default ``10``.
    maxls : int, optional
        Line-search steps per iteration; scipy's default ``20``.

    Returns
    -------
    UpdateClass
        A gradient ``UpdateClass`` with population size one.

    Raises
    ------
    ValueError
        If ``maxcor`` or ``maxls`` is below one.
    """
    bounded = (None, None) if bounded is None else tuple(bounded)
    return _stepped_update(
        "lbfgsb",
        lbfgsb_transform(maxcor=maxcor, maxls=maxls, bounded=bounded),
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


def slsqp_update(
    *,
    model: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
    opt_hash: int,
    bounded: tuple[float | None, float | None] | None = (None, None),
    stop_fn: StopFunction | None = None,
) -> UpdateClass:
    """Construct the ``UpdateClass`` behind ``optimizer="slsqp"``.

    scipy's SLSQP (``minimize(method="SLSQP")``): Kraft's sequential
    least-squares quadratic programming, an SQP method with a dense BFGS
    Hessian and an L1 merit line search. With no constraints beyond a
    box, each step is a bound-constrained least-squares subproblem. Its
    dense matrices make the cost per step grow steeply above about a
    thousand dimensions. It takes no hyperparameters.

    scipy is stepped one evaluation at a time (:func:`slsqp_transform`,
    ADR 0011), so the entry can sit in a switching menu and honours a
    ``stop_fn``. Without one, a run lasts its whole budget: once scipy
    stops, every remaining step re-evaluates its final point. Its
    ``ftol`` is zero (ADR 0003), so it stops on "positive directional
    derivative for linesearch".

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
        Stable hash stamped into ``OptHistory.update_step``.
    bounded : tuple[float | None, float | None] or None, optional
        Box handed to scipy, which keeps every iterate inside it (to a
        rounding error, which the loss's projection absorbs); the
        starting point is clipped into it. ``None``, or ``None`` on one
        side, leaves that side unbounded.
    stop_fn : StopFunction or None, optional
        Stopping function; ``None`` never stops.

    Returns
    -------
    UpdateClass
        A gradient ``UpdateClass`` with population size one.
    """
    bounded = (None, None) if bounded is None else tuple(bounded)
    return _stepped_update(
        "slsqp",
        slsqp_transform(bounded=bounded),
        model=model,
        loss_fn=loss_fn,
        pass_rng=pass_rng,
        opt_hash=opt_hash,
        bounded=bounded,
        stop_fn=stop_fn,
    )


#: The parts-path transform of each stepped entry, keyed as in the
#: registry. :func:`~l2co_optimizers._src.optimizer_parts.optimizer_parts`
#: builds it from the step's hyperparameters, unbounded.
STEPPED_SCIPY_TRANSFORMS: dict[
    str, Callable[..., optax.GradientTransformationExtraArgs]
] = {"lbfgsb": lbfgsb_transform, "slsqp": slsqp_transform}

scipy_stepped_mapping = {
    "lbfgsb": Partial(lbfgsb_update),
    "slsqp": Partial(slsqp_update),
}
