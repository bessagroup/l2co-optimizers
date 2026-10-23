"""What every switchable optimizer declares to the switching layer.

At a switch, l2co decides whether the incoming optimizer's own state
still holds where it is being sent (l2co ADR 0022). That needs four
things from every optimizer, whatever kind it is, and this module
builds them (ADR 0012):

* **where it is** -- its *point*: an ES's mean, a population's centroid,
  a trust region's centre. A gradient optimizer's point is the iterate,
  which the switching loop holds, so it declares none here;
* **what shape it has learned** -- how much wider its search is along
  some directions than along others, with the overall size removed
  (:class:`Shape`), plus the scale that shape pairs with;
* **how fast that shape adapts** -- its *evidence rates*, the fraction
  of the diagonal and of a direction one update replaces, taken from
  the optimizer's own learning rates;
* **what a restart keeps** -- for most optimizers nothing, for the
  trust-region optimizers the points they have already evaluated.

How each optimizer reads its scale stays in :mod:`state_transfer`.
Which kind of point, shape, rates and restart an optimizer has is
declared per optimizer on its
:class:`~l2co_optimizers._src.core.state_transfer.TransferSpec`; this
module only implements the kinds, so it names no optimizer itself.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import dataclasses
from collections.abc import Callable

# Third-party
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

# Local
from l2co_optimizers._src.core.state_transfer import (
    FAMILY_DISTRIBUTION,
    FAMILY_GRADIENT,
    FAMILY_POPULATION,
    transfer_spec_for,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "Shape",
    "SwitchPorts",
    "build_switch_ports",
]

#: A direction is reported only if its stretch differs from 1 by more
#: than this factor. Provisional: ADR 0012's slope test fixes it.
DIRECTION_THRESHOLD = 1.5

#: Eigenvalues below this fraction of the largest are floored before a
#: shape is formed: "not curved upward" is treated as "very flat"
#: (ADR 0012).
EIGENVALUE_FLOOR = 1e-6


class Shape(eqx.Module):
    """A search shape in the one unit every optimizer reads into.

    The shape is ``D (I + sum_i (s_i - 1) u_i u_iᵀ) D`` with determinant
    1 -- the geometric mean of its stretches is 1, so it carries no size
    -- and :attr:`scale` is the width it pairs with: ``scale²`` times the
    shape is the covariance-like spread the optimizer searches with.
    ``u_0`` is the widest direction relative to the diagonal and ``u_1``
    the narrowest; both are unit vectors after scaling by the diagonal.
    See ADR 0012.

    :attr:`scale` is a geometric-mean width. The scale readers in
    :mod:`state_transfer` report an RMS width instead, which on an
    ill-conditioned search can be orders of magnitude larger; only this
    one pairs with a determinant-1 shape.

    Attributes
    ----------
    scale : Float[Array, ""]
        Geometric-mean width the shape pairs with.
    diagonal : Float[Array, " d"]
        ``D``, a length-like factor per coordinate. All ones when the
        diagonal was not measured.
    directions : Float[Array, "2 d"]
        The widest and the narrowest direction.
    stretches : Float[Array, " 2"]
        Their stretches relative to the diagonal: ``>= 1`` for the
        widest, ``<= 1`` for the narrowest, exactly 1 where absent.
    has_diagonal : Bool[Array, ""]
        Whether the diagonal was measured.
    has_direction : Bool[Array, " 2"]
        Whether each direction was found.
    """

    scale: Float[Array, ""]
    diagonal: Float[Array, " d"]
    directions: Float[Array, "2 d"]
    stretches: Float[Array, " 2"]
    has_diagonal: Bool[Array, ""]
    has_direction: Bool[Array, " 2"]

    @classmethod
    def neutral(cls, num_dims: int, scale: Float[Array, ""]) -> Shape:
        """The shape with no preferred direction, at ``scale``.

        Parameters
        ----------
        num_dims : int
            Dimensionality.
        scale : Float[Array, ""]
            Width it pairs with.

        Returns
        -------
        Shape
            Every stretch equal to 1.
        """
        return cls(
            scale=jnp.asarray(scale, dtype=float),
            diagonal=jnp.ones(num_dims),
            directions=jnp.zeros((2, num_dims)),
            stretches=jnp.ones(2),
            has_diagonal=jnp.asarray(False),
            has_direction=jnp.zeros(2, dtype=bool),
        )

    def distance(self, offset: Float[Array, " d"]) -> Float[Array, ""]:
        """How far ``offset`` reaches, in this shape's own widths.

        The Mahalanobis distance in the shape's metric at :attr:`scale`,
        divided by ``sqrt(d)`` so that it is per coordinate, like a
        scale: for a neutral shape it is the RMS of ``offset`` over the
        scale. Along a direction the shape says is wide the same offset
        counts for less, across it for more (l2co ADR 0022).

        Parameters
        ----------
        offset : Float[Array, " d"]
            Flat displacement from where the shape was measured.

        Returns
        -------
        Float[Array, ""]
            Distance in widths.
        """
        y = jnp.ravel(offset) / (self.scale * self.diagonal)
        along = self.directions @ y
        # (I + sum (s - 1) u uᵀ)⁻¹ = I + sum (1/s - 1) u uᵀ, the u_i
        # being orthonormal.
        correction = jnp.where(
            self.has_direction, (1.0 / self.stretches - 1.0) * along**2, 0.0
        )
        squared = jnp.sum(y**2) + jnp.sum(correction)
        return jnp.sqrt(jnp.maximum(squared, 0.0) / y.shape[0])


def _from_widths(widths: Float[Array, " d"]) -> Shape:
    """A diagonal shape from per-coordinate widths.

    Works on the widths' logarithms and never squares them: a step size
    that has collapsed below about 1e-154 squares to zero in float64
    (Sep-CMA-ES gets there on a sharp ridge), and its width must still
    read as that step size, not as zero. A width of exactly zero reads
    as the smallest normal float, so its logarithm stays finite.
    """
    widths = jnp.maximum(widths, jnp.sqrt(EIGENVALUE_FLOOR) * jnp.max(widths))
    log_widths = jnp.log(jnp.maximum(widths, jnp.finfo(widths.dtype).tiny))
    log_mean = jnp.mean(log_widths)
    return Shape(
        scale=jnp.exp(log_mean),
        diagonal=jnp.exp(log_widths - log_mean),
        directions=jnp.zeros((2, widths.shape[0])),
        stretches=jnp.ones(2),
        has_diagonal=jnp.asarray(True),
        has_direction=jnp.zeros(2, dtype=bool),
    )


def _from_variances(variances: Float[Array, " d"]) -> Shape:
    """A diagonal shape from per-coordinate variances."""
    return _from_widths(jnp.sqrt(variances))


def _times_std(shape: Shape, std: Array) -> Shape:
    """``shape`` at ``std`` times its width.

    The readers form the shape from the optimizer's matrix alone and
    multiply its step size in afterwards, for the reason
    :func:`_from_widths` gives: ``std² C`` underflows long before
    ``std`` does.
    """
    std = jnp.reshape(jnp.asarray(std, dtype=float), ())
    return dataclasses.replace(shape, scale=std * shape.scale)


def _from_covariance(cov: Float[Array, "d d"]) -> Shape:
    """A diagonal plus the widest and narrowest direction of ``cov``.

    The directions are the extreme eigenvectors of the correlation
    matrix ``D⁻¹ cov D⁻¹``, so their stretches are relative to the
    diagonal; everything in between is dropped (ADR 0012).
    """
    num_dims = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    variances = jnp.diagonal(cov)
    variances = jnp.maximum(variances, EIGENVALUE_FLOOR * jnp.max(variances))
    root = jnp.sqrt(variances)
    corr = cov / (root[:, None] * root[None, :])
    values, vectors = jnp.linalg.eigh(corr)
    values = jnp.maximum(values, EIGENVALUE_FLOOR * values[-1])
    stretches = jnp.stack([values[-1], values[0]])
    directions = jnp.stack([vectors[:, -1], vectors[:, 0]])
    present = jnp.stack(
        [
            stretches[0] > DIRECTION_THRESHOLD,
            stretches[1] < 1.0 / DIRECTION_THRESHOLD,
        ]
    )
    # One dimension has one eigenvector; it cannot be both.
    if num_dims < 2:
        present = jnp.zeros(2, dtype=bool)
    stretches = jnp.where(present, stretches, 1.0)
    log_det = jnp.sum(jnp.log(variances)) + jnp.sum(jnp.log(stretches))
    log_mean = log_det / num_dims
    return Shape(
        scale=jnp.exp(0.5 * log_mean),
        diagonal=jnp.exp(0.5 * (jnp.log(variances) - log_mean)),
        directions=directions,
        stretches=stretches,
        has_diagonal=jnp.asarray(True),
        has_direction=present,
    )


def _std(opt_state: PyTree, num_dims: int) -> Array:
    """``std`` broadcast to one value per coordinate."""
    std = jnp.asarray(opt_state.std, dtype=float)
    return jnp.broadcast_to(std, (num_dims,))


def _read_crfmnes(opt_state: PyTree) -> Shape:
    """CR-FM-NES holds the layout natively: ``std² D (I + v vᵀ) D``.

    Its ``v`` can only widen a direction, so the narrowest is never
    reported.
    """
    D = jnp.asarray(opt_state.D, dtype=float)
    v = jnp.asarray(opt_state.v, dtype=float)
    num_dims = D.shape[0]
    root = D
    norm_sq = jnp.sum(v**2)
    stretch = 1.0 + norm_sq
    present = jnp.stack([stretch > DIRECTION_THRESHOLD, jnp.asarray(False)])
    stretches = jnp.where(present, jnp.stack([stretch, 1.0]), 1.0)
    unit = v / jnp.sqrt(jnp.maximum(norm_sq, 1e-300))
    log_det = 2.0 * jnp.sum(jnp.log(root)) + jnp.sum(jnp.log(stretches))
    log_mean = log_det / num_dims
    shape = Shape(
        scale=jnp.exp(0.5 * log_mean),
        diagonal=root / jnp.exp(0.5 * log_mean),
        directions=jnp.stack([unit, jnp.zeros(num_dims)]),
        stretches=stretches,
        has_diagonal=jnp.asarray(True),
        has_direction=present,
    )
    return _times_std(shape, opt_state.std)


def _read_cov(opt_state: PyTree) -> Shape:
    """CMA-ES: ``std² C``."""
    C = jnp.asarray(opt_state.C, dtype=float)
    return _times_std(_from_covariance(C), opt_state.std)


def _read_cov_diag(opt_state: PyTree) -> Shape:
    """Sep-CMA-ES: ``std² C`` with ``C`` a diagonal."""
    C = jnp.asarray(opt_state.C, dtype=float)
    return _times_std(_from_variances(C), opt_state.std)


def _read_mmt(opt_state: PyTree) -> Shape:
    """MA-ES: ``std² M Mᵀ``."""
    M = jnp.asarray(opt_state.M, dtype=float)
    return _times_std(_from_covariance(M @ M.T), opt_state.std)


def _read_btb(opt_state: PyTree) -> Shape:
    """xNES: ``std² Bᵀ B`` (it samples ``z @ B``)."""
    B = jnp.asarray(opt_state.B, dtype=float)
    return _times_std(_from_covariance(B.T @ B), opt_state.std)


def _read_std_diag(opt_state: PyTree) -> Shape:
    """SNES, SimpleES: one ``std`` per coordinate."""
    num_dims = jnp.asarray(opt_state.mean).shape[0]
    return _from_widths(_std(opt_state, num_dims))


def _read_iamalgam(opt_state: PyTree) -> Shape:
    """iAMaLGaM-full samples from ``C + std² I``."""
    C = jnp.asarray(opt_state.C, dtype=float)
    std = jnp.asarray(opt_state.std, dtype=float)
    # Divide the largest width out before squaring, as the other
    # readers do (twice by k, since k² may underflow).
    largest = jnp.maximum(std, jnp.sqrt(jnp.max(jnp.diagonal(C))))
    k = jnp.maximum(largest, jnp.finfo(largest.dtype).tiny)
    eye = jnp.eye(C.shape[0])
    # Both parts exactly zero: no shape at all, so a neutral one.
    cov = jnp.where(largest > 0, C / k / k + (std / k) ** 2 * eye, eye)
    return _times_std(_from_covariance(cov), k)


def _read_iamalgam_diag(opt_state: PyTree) -> Shape:
    """iAMaLGaM-univariate samples with widths ``std + sqrt(C)``."""
    C = jnp.asarray(opt_state.C, dtype=float)
    return _from_widths(_std(opt_state, C.shape[0]) + jnp.sqrt(C))


SHAPE_READERS: dict[str, Callable[[PyTree], Shape]] = {
    "crfmnes": _read_crfmnes,
    "cov": _read_cov,
    "cov_diag": _read_cov_diag,
    "mmt": _read_mmt,
    "btb": _read_btb,
    "std_diag": _read_std_diag,
    "iamalgam": _read_iamalgam,
    "iamalgam_diag": _read_iamalgam_diag,
}
"""Shape readers by kind, each verified against the algorithm's ``_ask``."""


def _scale_std(state: PyTree, factor: Array) -> PyTree:
    return state.replace(std=jnp.asarray(state.std, dtype=float) * factor)


def _make_neutralize(kind: str, fields: tuple[str, ...]) -> Callable:
    """``neutralize(old_state, fresh_state)`` for a shape of ``kind``.

    Takes the shape fields from the optimizer's own fresh ``init`` and
    rescales its ``std`` so the geometric-mean width stays the old
    state's. "Neutral" therefore means the optimizer's own starting
    shape at its current width, not a shape it never starts from:
    CR-FM-NES, for one, starts from a random ``v`` rather than zero, and
    its update divides by ``|v|``.
    """
    read = SHAPE_READERS[kind]
    # Where std multiplies the shape, the width ratio is the shapes'
    # ratio at unit std: it then survives a std that has collapsed to
    # zero, which ``0 / 0`` would not.
    std_times_shape = kind not in ("std_diag", "iamalgam", "iamalgam_diag")

    def _unit_std(state: PyTree) -> PyTree:
        return state.replace(std=jnp.ones_like(jnp.asarray(state.std, float)))

    def neutralize(old_state: PyTree, fresh_state: PyTree) -> PyTree:
        """Forget the shape, keep the width (l2co ADR 0022, arm 3a).

        Parameters
        ----------
        old_state : PyTree
            The optimizer's state, with the shape to forget.
        fresh_state : PyTree
            Its own ``init``, for the starting shape fields.

        Returns
        -------
        PyTree
            ``old_state`` with its starting shape, at its old width.
        """
        width = read(old_state).scale
        state = old_state.replace(
            **{name: getattr(fresh_state, name) for name in fields}
        )
        if kind in ("iamalgam", "iamalgam_diag"):
            # The width is std plus the sampled part, not std times it:
            # put all of it in std and none in the shape.
            state = state.replace(
                std=jnp.broadcast_to(width, jnp.shape(state.std)),
                C=jnp.zeros_like(state.C),
            )
            return state
        if std_times_shape:
            ratio = (
                read(_unit_std(old_state)).scale / read(_unit_std(state)).scale
            )
        else:
            ratio = width / read(state).scale
        return _scale_std(state, ratio)

    return neutralize


#: The fields that hold each kind of shape, as ``init`` sets them.
_SHAPE_FIELDS: dict[str, tuple[str, ...]] = {
    "crfmnes": ("D", "v"),
    "cov": ("C", "B", "D"),
    "cov_diag": ("C", "D"),
    "mmt": ("M",),
    "btb": ("B",),
    "std_diag": ("std",),
    "iamalgam": ("C",),
    "iamalgam_diag": ("C",),
}


def _point_mean(opt_state: PyTree) -> Array:
    return jnp.ravel(jnp.asarray(opt_state.mean, dtype=float))


def _point_centroid(opt_state: PyTree) -> Array:
    population = jnp.asarray(opt_state.population, dtype=float)
    return jnp.nanmean(population.reshape(population.shape[0], -1), axis=0)


def _point_center(opt_state: PyTree) -> Array:
    return jnp.ravel(jnp.asarray(opt_state.center, dtype=float))


POINT_READERS: dict[str, Callable[[PyTree], Array] | None] = {
    "mean": _point_mean,
    "centroid": _point_centroid,
    "center": _point_center,
    # The switching loop holds a gradient optimizer's iterate itself.
    "iterate": None,
    "none": None,
}
"""Point readers by kind; ``None`` means the optimizer holds no point."""

#: Point kind by family, unless the optimizer's spec says otherwise.
_DEFAULT_POINT = {
    FAMILY_DISTRIBUTION: "mean",
    FAMILY_POPULATION: "centroid",
    FAMILY_GRADIENT: "iterate",
}


def _make_restart(keeps: tuple[str, ...]) -> Callable:
    """``restart(old_state, fresh_state) -> state`` keeping ``keeps``."""

    def restart(old_state: PyTree, fresh_state: PyTree) -> PyTree:
        """Start afresh, keeping only what this optimizer declares.

        Parameters
        ----------
        old_state : PyTree
            The state the optimizer had before.
        fresh_state : PyTree
            Its own ``init`` at the restart point.

        Returns
        -------
        PyTree
            ``fresh_state`` with the kept fields taken from
            ``old_state``.
        """
        if not keeps:
            return fresh_state
        kept = {name: getattr(old_state, name) for name in keeps}
        if hasattr(fresh_state, "replace"):
            return fresh_state.replace(**kept)
        return dataclasses.replace(fresh_state, **kept)

    return restart


class SwitchPorts(eqx.Module):
    """One optimizer's declarations to the switching layer (ADR 0012).

    All fields are static.

    Attributes
    ----------
    read_point : Callable or None
        ``read_point(opt_state) -> x``, the flat point the optimizer is
        at. ``None`` for a gradient optimizer, whose point is the
        iterate the switching loop holds, and for one that holds no
        point at all.
    read_shape : Callable or None
        ``read_shape(opt_state) -> Shape``. ``None`` when the optimizer
        has no shape of its own, or when no evidence rate is known for
        it (ADR 0012: no rate, no reader).
    shape_rates : tuple[float, float] or None
        Fraction of the diagonal and of a direction one update replaces.
        Set exactly when :attr:`read_shape` is.
    restart : Callable
        ``restart(old_state, fresh_state) -> state``. Given the
        optimizer's fresh ``init`` at the restart point, keeps what the
        optimizer declares worth keeping across a restart.
    restart_keeps : tuple[str, ...]
        The state fields :attr:`restart` keeps.
    neutralize : Callable or None
        ``neutralize(old_state, fresh_state) -> state``: forget the
        learned shape, keep the width. Takes the shape fields from the
        optimizer's fresh ``init`` and rescales its scale so the
        geometric-mean width is unchanged. Set exactly when
        :attr:`read_shape` is.
    """

    read_point: Callable | None = eqx.field(static=True)
    read_shape: Callable | None = eqx.field(static=True)
    shape_rates: tuple[float, float] | None = eqx.field(static=True)
    restart: Callable = eqx.field(static=True)
    restart_keeps: tuple[str, ...] = eqx.field(static=True, default=())
    neutralize: Callable | None = eqx.field(static=True, default=None)


def build_switch_ports(
    name: str, family: str, es_params: PyTree | None = None
) -> SwitchPorts:
    """Bind one optimizer's switch ports at construction time.

    Parameters
    ----------
    name : str
        Registry name.
    family : str
        The optimizer's family.
    es_params : PyTree or None, optional
        The evosax parameters the optimizer was built with. Its evidence
        rates are read from these, so an optimizer with a shape reader
        needs them.

    Returns
    -------
    SwitchPorts
        The optimizer's point, shape, evidence rates and restart.

    Raises
    ------
    ValueError
        If the optimizer has a shape reader but ``es_params`` is missing.
    """
    spec = transfer_spec_for(name, family)
    point = spec.point if spec.point is not None else _DEFAULT_POINT[family]

    read_shape, rates, neutralize = None, None, None
    if spec.shape != "none":
        if es_params is None:
            raise ValueError(
                f"{name!r} has a shape reader ({spec.shape!r}) but no "
                f"es_params to read its evidence rates from"
            )
        diagonal_fields, direction_fields = spec.shape_rates
        rates = (
            float(sum(getattr(es_params, f) for f in diagonal_fields)),
            float(sum(getattr(es_params, f) for f in direction_fields)),
        )
        read_shape = SHAPE_READERS[spec.shape]
        neutralize = _make_neutralize(spec.shape, _SHAPE_FIELDS[spec.shape])

    return SwitchPorts(
        read_point=POINT_READERS[point],
        read_shape=read_shape,
        shape_rates=rates,
        restart=_make_restart(spec.restart_keeps),
        restart_keeps=spec.restart_keeps,
        neutralize=neutralize,
    )
