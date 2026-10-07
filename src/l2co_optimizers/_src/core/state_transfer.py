"""What one optimizer can hand the next when control passes to it.

The handshake -- l2co's handshake policy (l2co ADR 0009), which lives
with the rest of the switching layer in l2co (l2co ADR 0019) -- answers
*which population* the incoming optimizer sees and *whether* its internal
state survives. It carries no information **out of** the outgoing
optimizer: the incoming one is handed a point and, at best, told to keep
or discard its own accumulated state. Everything the outgoing optimizer
learned about the landscape is discarded at every switch.

This module is the channel for that information. It is deliberately
**not** a pairwise ``(outgoing, incoming)`` table. Each optimizer gets
one *reader* that projects its own state onto a small canonical bundle,
and one *writer* that folds a bundle into its own state -- ``2N``
adapters rather than ``N²``. The justification is maintainability
(``2N`` hand-written adapters against ``N²`` for a registry of 59),
compile time and code size (a pairwise dispatch would trace ``N²``
branches), and redundancy (most of those ``N²`` cells would be derived
from the same handful of statistics anyway). It is explicitly **not** a
per-tick runtime argument: rl2co's ``map_unroll`` and
:meth:`UpdateClass.batch_run_sequential` both execute only
the branch actually selected, so per-tick cost does not scale with the
branch count.

What is carried
---------------
Only **scale** -- a single positive scalar ``sigma`` -- plus the location
it belongs to. That is not a placeholder for a richer bundle so much as
the slot that carries almost all of the available signal: over the
registry, 27 optimizers hold an explicit ``std`` and 20 hold a diagonal
second moment, and on a menu mixing gradient-based and
distribution-based optimizers the *initial* scales can differ by three
orders of magnitude (``rprop`` at ``learning_rate=1e-3`` against an ES at
``std_init=1.0``). Shape and direction are deliberately absent; see
``l2co ADR 0013``.

The governing principle for writes is that **scale goes stale and shape
does not**. An optimizer that has not run for many generations holds a
``std`` that is wrong about the current search, but its *relative*
per-coordinate structure is still informative. So a write never
overwrites a per-coordinate quantity with a flat scalar -- it **rescales**
it so its RMS matches the incoming ``sigma`` while the relative structure
survives. Writing ``sigma`` flat into ``rprop``'s ``step_sizes`` would
destroy exactly the per-coordinate ratchet that makes ``rprop`` work.

Who calls this
--------------
The population half of a switch stays with the caller, keyed on
:attr:`TransferSpec.own_ask` -- a per-optimizer property, not a family
one (``l2co ADR 0014``):

* ``own_ask`` -- the caller writes location and scale through
  :func:`build_transfer_fns`'s writer, then asks the incoming optimizer
  for its **own** population. Two things need this, for one reason.
  CR-FM-NES and seven relatives cache their draws in state and consume
  them on the next ``tell``, so substituting a foreign population
  silently credits each candidate's fitness to a direction that produced
  a different point (``l2co ADR 0012``). The mutation GAs cache no
  draws, but their next generation is *scored against a baseline this
  handshake just wrote* -- MR15-GA's 1/5 rule, GESMR-GA's per-group
  delta, SAMR-GA's elite sort -- so a foreign population is measured
  against a baseline it was not drawn from. Both are the same hazard:
  the optimizer's next update reads state the handshake touched.
  Regenerating through its own ``ask`` makes the two consistent by
  construction.
* otherwise -- the caller keeps the existing layout: ``best_one`` for a
  population-based algorithm whose stored population *is* the state, and
  ``best`` for the unit-population gradient family.

Reader and writer are built at optimizer-construction time rather than
looked up per call, because both need hyperparameters that are not
recoverable from the state: ``rprop``'s ``min_step_size`` /
``max_step_size`` clip bounds, the learning rate of every optax
optimizer (``scale_by_learning_rate`` carries an ``EmptyState``), and the
ravel function an evosax algorithm uses to flatten a solution pytree.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float, PyTree

# Local
from l2co_optimizers._src.core.typing import (
    TransferReadFunction,
    TransferWriteFunction,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "CONF_ABSENT",
    "CONF_ESTIMATED",
    "CONF_EXACT",
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_GRADIENT",
    "FAMILY_POPULATION",
    "TRANSFER_OVERRIDES",
    "TransferBundle",
    "TransferSpec",
    "build_transfer_fns",
    "transfer_spec_for",
]


#                                                            Confidence levels
# =============================================================================

CONF_ABSENT = 0.0
"""The outgoing optimizer holds no scale information. Writers skip."""

CONF_ESTIMATED = 0.5
"""Scale inferred from samples or from an unrecognised shape field.

Written the same way an exact value is -- the distinction is carried so a
later blend rule can weigh it, and so ``CONF_ABSENT`` stays the only
level that changes behaviour.
"""

CONF_EXACT = 1.0
"""Scale read directly off a field that *is* the optimizer's scale."""


#                                                          Optimizer families
# =============================================================================

FAMILY_DISTRIBUTION = "distribution"
"""Holds an explicit search distribution; regenerates its own population."""

FAMILY_POPULATION = "population"
"""The stored population is the state."""

FAMILY_GRADIENT = "gradient"
"""Unit population, iterate plus a preconditioner."""

FAMILY_DERIVATIVE_FREE = "derivative_free"
"""Unit population; an iterate plus a derivative-free local model.

The model is an interpolation set (COBYQA) or a set of search
directions (Powell). Only plain-run entries hold this family (ADR
0002), so it has no transfer spec: :func:`transfer_spec_for` refuses it.
"""

FAMILIES = (
    FAMILY_DISTRIBUTION,
    FAMILY_POPULATION,
    FAMILY_GRADIENT,
    FAMILY_DERIVATIVE_FREE,
)


#                                                              The bundle
# =============================================================================


class TransferBundle(eqx.Module):
    """What crosses a switch.

    Every reader must produce this same structure: it is the output of a
    ``lax.switch`` over the outgoing optimizer's index, so all branches
    must agree on layout leaf for leaf.

    Attributes
    ----------
    x : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    f : Float[Array, ""]
        Loss at ``x``.
    sigma : Float[Array, ""]
        Characteristic scale of the outgoing optimizer's search: the RMS
        per-coordinate standard deviation for a sampler, the RMS step
        length for a stepper. Meaningless unless ``conf_sigma`` is above
        :data:`CONF_ABSENT`.
    conf_sigma : Float[Array, ""]
        One of :data:`CONF_ABSENT`, :data:`CONF_ESTIMATED`,
        :data:`CONF_EXACT`.
    moves_iterate : Bool[Array, ""]
        Whether this handshake is actually *moving* the incoming
        optimizer, i.e. whether ``x`` is somewhere it did not already
        know about. ``True`` unless the caller says otherwise, which is
        the conservative default: a handshake that does not know moves
        the iterate, and every write below is written for that case.

        The writers use it for the half of their work that is justified
        by the jump rather than by the scale going stale — re-centring a
        ``mean``, zeroing a cumulation path, rebasing an elite, clearing
        Rprop's ``prev_updates``. All of those are *repairs for having
        moved*. When nothing moved they are not conservative, they are
        destructive: they discard adaptation the incoming optimizer
        earned, to protect it from a jump that is not happening. The
        scale write is unaffected — an outgoing optimizer that found
        nothing better may still have learned the local scale.

        This is a runtime value rather than a static flag because the
        caller's predicate is traced (it compares the bundle's loss
        against the incoming optimizer's own best).
    """

    x: PyTree
    f: Float[Array, ""]
    sigma: Float[Array, ""]
    conf_sigma: Float[Array, ""]
    moves_iterate: Bool[Array, ""] = eqx.field(
        default_factory=lambda: jnp.asarray(True)
    )

    @classmethod
    def absent(cls, x: PyTree, f: Float[Array, ""]) -> TransferBundle:
        """Build a bundle carrying location only, no scale.

        Parameters
        ----------
        x : PyTree
            Best parameters seen so far.
        f : Float[Array, ""]
            Loss at ``x``.

        Returns
        -------
        TransferBundle
            Bundle whose ``conf_sigma`` is :data:`CONF_ABSENT`.
        """
        return cls(
            x=x,
            f=jnp.asarray(f, dtype=float),
            sigma=jnp.asarray(0.0, dtype=float),
            conf_sigma=jnp.asarray(CONF_ABSENT, dtype=float),
        )


#                                                              Sigma readers
# =============================================================================


def _rms(x: PyTree) -> Array:
    """Root-mean-square over every leaf of ``x``, NaN-safe over all-NaN.

    ``x`` is usually a bare array, but a task whose decision vector is
    itself a pytree (e.g. an ``eqx.Module`` rather than a flat array)
    makes any optimizer-state field that mirrors it -- Rprop's
    ``step_sizes``, L-BFGS's ``diff_params_memory`` -- a pytree of the
    same shape too. Flattening across every leaf reduces to the same
    scalar as before whenever ``x`` already is one array.
    """
    squares = jnp.concatenate(
        [
            jnp.ravel(jnp.square(jnp.asarray(leaf, dtype=float)))
            for leaf in jax.tree.leaves(x)
        ]
    )
    return jnp.sqrt(jnp.nanmean(squares))


def _shape_factor_sq(opt_state: PyTree) -> tuple[Array, float]:
    """Per-coordinate variance factor implied by an ES's shape fields.

    An evosax distribution-based algorithm samples
    ``x = mean + std * T(z)`` for some shape transform ``T`` held in the
    state. The RMS per-coordinate standard deviation is therefore
    ``std * sqrt(mean(diag(T Tᵀ)))``, and this returns that
    ``mean(diag(T Tᵀ))`` together with the confidence it deserves.

    Recognised, each verified against the algorithm's ``_ask``:

    * ``D`` and ``v`` -- CR-FM-NES, ``Cov = std² D (I + v vᵀ) D``, so the
      diagonal is ``D² (1 + v²)`` (``cr_fm_nes.py:176-184``).
    * ``C`` one-dimensional -- Sep-CMA-ES, ``x = mean + std * (sqrt(C) *
      z)`` (``sep_cma_es.py:66-78``).
    * ``C`` two-dimensional -- CMA-ES, ``Cov = std² C``
      (``cma_es.py:162-176``).
    * neither -- an identity transform, e.g. SNES or OpenES, where the
      per-coordinate scale lives entirely in ``std``
      (``snes.py:76-84``).

    Anything else (``maes``/``lmmaes``'s ``M``, ``xnes``'s ``B``) falls
    back to ``1.0`` and downgrades the confidence rather than guessing a
    relation.

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, float]
        ``(factor_sq, confidence)``.
    """
    D = getattr(opt_state, "D", None)
    v = getattr(opt_state, "v", None)
    if D is not None and v is not None:
        D = jnp.asarray(D, dtype=float)
        v = jnp.asarray(v, dtype=float)
        return jnp.nanmean(jnp.square(D) * (1.0 + jnp.square(v))), CONF_EXACT

    C = getattr(opt_state, "C", None)
    if C is not None:
        C = jnp.asarray(C, dtype=float)
        if C.ndim == 1:
            return jnp.nanmean(C), CONF_EXACT
        if C.ndim == 2:
            return jnp.nanmean(jnp.diagonal(C)), CONF_EXACT

    # No shape field at all: the transform is the identity and ``std``
    # already carries the whole per-coordinate scale.
    if not any(
        getattr(opt_state, name, None) is not None
        for name in ("M", "B", "D", "v", "C")
    ):
        return jnp.asarray(1.0), CONF_EXACT

    return jnp.asarray(1.0), CONF_ESTIMATED


def read_sigma_distribution(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the sampling scale off a distribution-based ES state.

    ``std`` is a scalar for most algorithms and a per-coordinate vector
    for a few (SNES); ``_rms`` handles both, and the shape fields supply
    the remaining factor.

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    std = getattr(opt_state, "std", None)
    if std is None:
        return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

    factor_sq, conf = _shape_factor_sq(opt_state)
    sigma = _rms(std) * jnp.sqrt(factor_sq)
    return _finite_or_absent(sigma, conf)


def read_sigma_population(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the search scale off a population-based state.

    Three sources, in order of how directly they mean "scale":
    an explicit mutation ``std`` (the GA family), the RMS ``velocity``
    (PSO -- its per-iteration step), and otherwise the RMS
    per-coordinate spread of the stored population (DE, which holds
    neither).

    Parameters
    ----------
    opt_state : PyTree
        The algorithm's own state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    std = getattr(opt_state, "std", None)
    if std is not None:
        return _finite_or_absent(_rms(std), CONF_EXACT)

    velocity = getattr(opt_state, "velocity", None)
    if velocity is not None:
        # Zero at init, before any ``ask`` has moved a particle -- that
        # is absence of information, not a scale of zero.
        return _finite_or_absent(_rms(velocity), CONF_ESTIMATED)

    population = getattr(opt_state, "population", None)
    if population is not None:
        population = jnp.asarray(population, dtype=float)
        spread = jnp.sqrt(jnp.nanmean(jnp.nanvar(population, axis=0)))
        return _finite_or_absent(spread, CONF_ESTIMATED)

    return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)


#: Field names optax uses for the first moment, most specific first.
_MOMENTUM_NAMES = ("mu", "m", "trace")

#: Field names optax uses for the second moment that *divides* the step,
#: most specific first. ``adan``'s denominator is ``n`` -- its ``v`` is a
#: moment of gradient *differences* and appears in the numerator
#: (``transform.py:653-680``) -- so ``n`` must be tried before ``v``,
#: which is ``adafactor``'s unfactored second moment.
_SECOND_MOMENT_NAMES = ("nu", "n", "v", "sum_of_squares")

#: Field names optax uses for the update counter.
_COUNT_NAMES = ("count", "t")


def _get_first(opt_state: PyTree, names: tuple[str, ...]):
    """Return the first present field among ``names``, with its name.

    optax spells the same quantity differently across transforms
    (``mu``/``m``/``trace``, ``nu``/``n``/``v``), so the readers and
    writers resolve by alias rather than assuming Adam's naming.

    Parameters
    ----------
    opt_state : PyTree
        An optax optimizer state, possibly a chain.
    names : tuple[str, ...]
        Candidate field names, most specific first.

    Returns
    -------
    tuple[Array | None, str | None]
        ``(value, name)``, or ``(None, None)`` if none are present.
    """
    for name in names:
        value = optax.tree.get(opt_state, name)
        if value is not None:
            return value, name
    return None, None


def make_read_sigma_gradient(learning_rate: float) -> Callable:
    """Build a reader for an optax second-moment optimizer.

    The transferable scale is the RMS of the *step* the optimizer would
    take, ``learning_rate * mu / (sqrt(nu) + eps)``. Neither the learning
    rate nor the bias-correction coefficients are recoverable from the
    state -- ``scale_by_learning_rate`` carries an ``EmptyState`` -- so
    the rate is closed over here and bias correction is skipped, which is
    why the result is only :data:`CONF_ESTIMATED`.

    Parameters
    ----------
    learning_rate : float
        The rate the optimizer was constructed with.

    Returns
    -------
    Callable
        ``reader(opt_state) -> (sigma, conf_sigma)``.
    """

    def read(opt_state: PyTree) -> tuple[Array, Array]:
        momentum, _ = _get_first(opt_state, _MOMENTUM_NAMES)
        if momentum is None:
            # A preconditioner without a stored step (RMSProp, Adagrad,
            # Adafactor) cannot report a scale: the step it would take
            # depends on a gradient magnitude nothing retained. Such
            # optimizers are receivers only -- see
            # :func:`make_write_gradient`.
            return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

        step = jnp.asarray(momentum, dtype=float)
        second, _ = _get_first(opt_state, _SECOND_MOMENT_NAMES)
        if second is not None:
            step = step / (jnp.sqrt(jnp.asarray(second, dtype=float)) + 1e-8)

        sigma = float(learning_rate) * _rms(step)
        sigma, conf = _finite_or_absent(sigma, CONF_ESTIMATED)

        count, _ = _get_first(opt_state, _COUNT_NAMES)
        if count is not None:
            # Before the first update the moments are zeros, so the step
            # RMS is zero and means nothing.
            conf = jnp.where(jnp.asarray(count) > 0, conf, CONF_ABSENT)
        return sigma, conf

    return read


def read_sigma_rprop(opt_state: PyTree) -> tuple[Array, Array]:
    """Read the scale off Rprop's per-coordinate step sizes.

    ``step_sizes`` *is* a length, which makes Rprop the one optimizer
    whose scale field needs no unit conversion. Its provenance is a
    success-history ratchet rather than a curvature estimate, but the
    quantity transferred here is scale, and for scale it is exact.

    Parameters
    ----------
    opt_state : PyTree
        Rprop's optax state.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf_sigma)``.
    """
    step_sizes = optax.tree.get(opt_state, "step_sizes")
    if step_sizes is None:
        return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)
    return _finite_or_absent(_rms(step_sizes), CONF_EXACT)


def make_read_sigma_lbfgs(
    scale_init_precond: bool = True,
    learning_rate: float | None = None,
) -> Callable:
    """Build the reader for L-BFGS.

    Two sources, because the obvious one is blind for exactly the visit
    length a switching policy asks for most often.

    **After two or more steps** the scale is the RMS of the most recent
    accepted step, which ``diff_params_memory`` holds directly. The ring
    index is ``(count - 2) % memory_size``, which is **not** optax's own
    expression even though it names the same slot. Optax writes to
    ``prev_memory_idx = (state.count - 1) % memory_size``
    (``transform.py:1734``) and *then* returns ``count + 1`` (``:1798``),
    so its ``state.count`` is the pre-increment value while a reader
    holding the returned state sees the post-increment one -- one further
    along. Copying the expression across that frame shift reads the next,
    not-yet-written slot: zeros until the ring fills, and a step one
    generation stale afterwards. Verified empirically -- after ``n``
    steps ``count == n`` and rows ``0 .. n-2`` are populated.

    **After exactly one step** there is no difference to read, and the
    old reader reported ``CONF_ABSENT``. That is not a rare corner:
    ``lbfgs`` is the one optimizer tabulated ``reset``, so its ``count``
    restarts on every switch *into* it, and "run lbfgs for 1 iteration"
    is one of the budget choices an rl2co policy has. The step it took
    is still recoverable, because every factor is in the state or closed
    over here: ``ScaleByLBFGSState.updates`` is the raw gradient
    ``g(x_0)`` (``transform.py:1799`` stores the argument, not the
    preconditioned direction), the memory was empty so the
    preconditioner was the capped reciprocal ``min(1, 1/||g||)``
    (``:1777-1783``), the chain's fixed multiplier is ``1`` for
    ``learning_rate=None`` and ``lr`` otherwise (``alias.py:2756-2759``),
    and the linesearch state's ``learning_rate`` is the stepsize it
    accepted. So ``sigma = alpha * lr * gamma * rms(g)``, exactly. A
    failed linesearch writes ``learning_rate = 0``, which
    :func:`_finite_or_absent` already downgrades to absence.

    The one configuration this cannot reconstruct is a *schedule* passed
    as ``learning_rate``: the multiplier then depends on a step count
    this reader cannot resolve to a number. There the first-step path is
    disabled rather than guessed -- reporting a scale that is wrong by an
    unknown factor is worse than reporting none -- and the memory path is
    unaffected.

    Parameters
    ----------
    scale_init_precond : bool, optional
        Whether the optimizer was built with the capped-reciprocal
        identity preconditioner. Defaults to True, as ``optax.lbfgs``
        does.
    learning_rate : float or None, optional
        The fixed multiplier the optimizer was built with. ``None``
        (the default) means the chain uses ``scale(-1.0)``.

    Returns
    -------
    Callable
        ``reader(opt_state) -> (sigma, conf_sigma)``.
    """
    if learning_rate is None:
        lr_mult, first_step_readable = 1.0, True
    elif isinstance(learning_rate, (int, float)):
        lr_mult, first_step_readable = abs(float(learning_rate)), True
    else:
        # A schedule. Its value at step 0 is not resolvable here.
        lr_mult, first_step_readable = 1.0, False

    def _first_step(opt_state: PyTree) -> tuple[Array, Array]:
        """Reconstruct the very first step's RMS length."""
        updates = optax.tree.get(opt_state, "updates")
        alpha = optax.tree.get(opt_state, "learning_rate")
        if updates is None or alpha is None:
            return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

        flat = jnp.concatenate(
            [
                jnp.ravel(jnp.asarray(leaf, dtype=float))
                for leaf in jax.tree.leaves(updates)
            ]
        )
        gamma = (
            jnp.minimum(1.0, 1.0 / optax.tree.norm(updates))
            if scale_init_precond
            else jnp.asarray(1.0)
        )
        sigma = jnp.abs(jnp.asarray(alpha, dtype=float)) * lr_mult
        return _finite_or_absent(sigma * gamma * _rms(flat), CONF_EXACT)

    def read(opt_state: PyTree) -> tuple[Array, Array]:
        """Read the scale off L-BFGS's most recent accepted step."""
        memory = optax.tree.get(opt_state, "diff_params_memory")
        count = optax.tree.get(opt_state, "count")
        if memory is None or count is None:
            return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)

        # ``diff_params_memory`` mirrors ``params``' own pytree structure
        # -- for a flat-array task this is one array, but a task whose
        # decision vector is itself a pytree makes it one ring buffer per
        # leaf, all sharing the same leading (memory) axis.
        memory_size = jnp.asarray(jax.tree.leaves(memory)[0]).shape[0]
        count = jnp.asarray(count)
        prev = (count - 2) % memory_size
        last_step = jax.tree.map(
            lambda leaf: jnp.take(leaf, prev, axis=0), memory
        )

        sigma, conf = _finite_or_absent(_rms(last_step), CONF_EXACT)
        if not first_step_readable:
            return sigma, jnp.where(count > 1, conf, CONF_ABSENT)

        first_sigma, first_conf = _first_step(opt_state)
        settled = count > 1
        return (
            jnp.where(settled, sigma, first_sigma),
            jnp.where(
                settled,
                conf,
                jnp.where(count == 1, first_conf, CONF_ABSENT),
            ),
        )

    return read


read_sigma_lbfgs = make_read_sigma_lbfgs()
"""The L-BFGS reader for ``optax.lbfgs``'s own defaults."""


def read_sigma_absent(opt_state: PyTree) -> tuple[Array, Array]:
    """Report no scale information, for optimizers that hold none.

    Parameters
    ----------
    opt_state : PyTree
        Ignored.

    Returns
    -------
    tuple[Array, Array]
        ``(0.0, CONF_ABSENT)``.
    """
    del opt_state
    return jnp.asarray(0.0), jnp.asarray(CONF_ABSENT)


def _finite_or_absent(sigma, conf) -> tuple[Array, Array]:
    """Downgrade a non-finite or non-positive scale to absence.

    A zero scale is what an uninitialised velocity or second moment
    looks like, and writing it would freeze the incoming optimizer; a
    NaN would poison it. Both are absence of information.

    Parameters
    ----------
    sigma : Array or float
        Candidate scale.
    conf : Array or float
        Confidence the reader would otherwise claim.

    Returns
    -------
    tuple[Array, Array]
        ``(sigma, conf)`` with ``conf`` forced to :data:`CONF_ABSENT`
        where ``sigma`` is not finite and positive.
    """
    sigma = jnp.asarray(sigma, dtype=float)
    usable = jnp.isfinite(sigma) & (sigma > 0.0)
    return (
        jnp.where(usable, sigma, 0.0),
        jnp.where(usable, jnp.asarray(conf, dtype=float), CONF_ABSENT),
    )


def _all_finite(tree: PyTree) -> Array:
    """Whether every float leaf of ``tree`` is finite.

    Parameters
    ----------
    tree : PyTree
        Tree to check; non-float leaves are ignored.

    Returns
    -------
    Array
        Scalar boolean. A tree with no float leaves reports ``False`` --
        nothing usable is present.
    """
    leaves = [
        jnp.asarray(leaf)
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
    ]
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(x.astype(float))) for x in leaves])
    )


def _usable_sigma(bundle: TransferBundle) -> Array:
    """Whether ``bundle.sigma`` may be written.

    Re-checks finiteness rather than trusting ``conf_sigma`` alone. Every
    reader here routes through :func:`_finite_or_absent`, so a
    non-finite ``sigma`` paired with a confident ``conf_sigma`` cannot
    arise from this module -- but a hand-built bundle or a future reader
    that forgets the helper would poison the receiver silently, and the
    check costs one comparison.

    Parameters
    ----------
    bundle : TransferBundle
        The bundle being written.

    Returns
    -------
    Array
        Scalar boolean gate.
    """
    return (
        (bundle.conf_sigma > CONF_ABSENT)
        & jnp.isfinite(bundle.sigma)
        & (bundle.sigma > 0.0)
    )


def _jumping(bundle: TransferBundle) -> Array:
    """Whether the incoming optimizer is being moved somewhere new.

    Gates the writes that exist to *repair a jump* rather than to
    refresh a stale scale: re-centring a ``mean``, zeroing a cumulation
    path, rebasing an elite, clearing Rprop's ``prev_updates``. When
    nothing moved those are not conservative but destructive -- they
    discard adaptation the incoming optimizer earned, to protect it from
    a jump that is not happening.

    Finiteness of ``x`` is folded in for the same reason
    :func:`_usable_sigma` re-checks ``sigma``: a non-finite point is not
    a destination, and re-centring on one kills the receiver for the
    rest of the episode.

    Parameters
    ----------
    bundle : TransferBundle
        The bundle being written.

    Returns
    -------
    Array
        Scalar boolean gate.
    """
    return jnp.asarray(bundle.moves_iterate) & _all_finite(bundle.x)


#                                                              State writers
# =============================================================================


def _blend(use: Array, new: Array, old: Array) -> Array:
    """Select ``new`` where ``use`` holds, elementwise, keeping dtype."""
    return jnp.where(use, new, old).astype(jnp.asarray(old).dtype)


def _rescale_to(values: PyTree, sigma: Array, use: Array) -> PyTree:
    """Scale ``values`` so their RMS becomes ``sigma``.

    The relative structure across coordinates is preserved -- this is the
    "scale goes stale, shape does not" rule. A degenerate input (all
    zeros, so no relative structure to preserve) is left alone rather
    than divided by zero. ``values`` may be a pytree rather than a bare
    array (see :func:`_rms`); the shared rescale factor is one scalar
    computed over every leaf together, then applied leaf-wise so the
    result keeps ``values``' own structure.

    Parameters
    ----------
    values : PyTree
        Per-coordinate quantity to rescale.
    sigma : Array
        Target RMS.
    use : Array
        Boolean gate; where false, ``values`` passes through.

    Returns
    -------
    PyTree
        Rescaled values, same structure as ``values``.
    """
    current = _rms(values)
    safe = jnp.isfinite(current) & (current > 0.0)
    factor = jnp.where(safe, sigma / jnp.where(safe, current, 1.0), 1.0)
    use = use & safe
    return jax.tree.map(
        lambda leaf: _blend(
            use,
            jnp.asarray(leaf, dtype=float) * factor,
            jnp.asarray(leaf, dtype=float),
        ),
        values,
    )


def make_write_distribution(ravel_fn: Callable) -> Callable:
    """Build the writer for a distribution-based ES.

    Writes the handed-over location into ``mean``, sets the scale, and
    zeros the cumulation paths. Shape fields (``C``, ``D``, ``v``, ``B``,
    ``M``) are deliberately **kept**: they describe the geometry of the
    landscape, which does not go stale the way scale does.

    Zeroing ``p_std`` / ``p_c`` is the same reasoning restart schemes
    (IPOP/BIPOP) use. A cumulation path is calibrated against the
    expected step length under the algorithm's *own* distribution, so
    carrying one across a jump to a foreign iterate corrupts step-size
    control; keeping the covariance while resetting the paths is the
    standard restart.

    ``mean`` is stored **raveled** -- evosax flattens solutions
    internally (``distribution_based/base.py:60-69``) -- so the bundle's
    pytree ``x`` has to go through the algorithm's own ravel function.

    Parameters
    ----------
    ravel_fn : Callable
        The algorithm's ``_ravel_solution``.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        updates: dict[str, Array] = {}

        # A non-finite location is the one input that cannot be
        # rescaled, clipped or downgraded into something usable: it
        # would set ``mean`` to NaN, every ``ask`` from that
        # distribution returns NaN, every loss is NaN, and no later
        # ``tell`` can recover it -- the optimizer is dead for the rest
        # of the episode, including after switching away and back, since
        # the ``RunState`` persists. So the fallback is to keep the
        # optimizer's own mean and let it carry on from where it was.
        use_x = _jumping(bundle)
        mean = getattr(opt_state, "mean", None)
        if mean is not None:
            updates["mean"] = _blend(
                use_x,
                jnp.asarray(ravel_fn(bundle.x), dtype=float),
                jnp.asarray(mean, dtype=float),
            )

        use = _usable_sigma(bundle)
        std = getattr(opt_state, "std", None)
        if std is not None:
            std = jnp.asarray(std)
            # ``sigma`` is an RMS per-coordinate standard deviation, but
            # ``std`` is only one factor of that: the target keeps its own
            # shape fields, which contribute ``sqrt(factor_sq)``. Divide
            # that out so the *resulting* sampling scale is ``sigma``,
            # making the writer the exact inverse of the reader.
            factor_sq, _ = _shape_factor_sq(opt_state)
            safe = jnp.isfinite(factor_sq) & (factor_sq > 0.0)
            target = bundle.sigma / jnp.sqrt(jnp.where(safe, factor_sq, 1.0))
            if std.ndim == 0:
                updates["std"] = _blend(use, target, std)
            else:
                # Per-coordinate std (SNES): rescale, never flatten.
                updates["std"] = _rescale_to(std, target, use)

        # Gated on the jump, not on the scale: a cumulation path is
        # calibrated against the algorithm's own past steps, so it is
        # invalid *because the iterate moved*. If it did not, the path
        # is the optimizer's own hard-won step-size control and zeroing
        # it throws that away.
        for path in ("p_std", "p_c"):
            value = getattr(opt_state, path, None)
            if value is not None:
                updates[path] = _blend(
                    use_x, jnp.zeros_like(value), jnp.asarray(value)
                )

        return opt_state.replace(**updates) if updates else opt_state

    return write


def make_write_population(repair_baseline: bool) -> Callable:
    """Build the writer for a population-based algorithm.

    Scale lands in whichever field carries it: an explicit mutation
    ``std`` (GA family) or the ``velocity`` (PSO, rescaled so its RMS
    matches while each particle's direction survives). The stored
    population is left to the caller's ``best_one`` handshake, or to the
    optimizer's own ``ask`` where :attr:`TransferSpec.own_ask` is set.

    ``repair_baseline`` additionally overwrites the stored
    ``population`` / ``fitness`` with the handed-over best. That is
    needed where the algorithm adapts against its *own stored* baseline
    on the next generation: MR15-GA's 1/5 rule is
    ``mean(fitness < state.fitness)`` (``mr15_ga.py:96``), so a stale
    elite from an early generation makes a population drawn around a much
    better point look like a near-total success and **doubles** the
    mutation scale exactly when it should shrink -- undoing the scale
    write one generation later. Rebasing on the handed-over best makes
    the rule ask the honest question. It also keeps the algorithm's
    ``(µ+λ)`` selection over ``concat(new, old)`` coherent, which
    overwriting ``fitness`` alone would not: stale rows would keep their
    positions while claiming the new fitness.

    Writing ``fitness = bundle.f`` **claims a measurement**: it asserts
    that a re-evaluation of ``bundle.x`` would return ``bundle.f``. On a
    stochastic loss it will not, because ``best_loss`` is a running
    *minimum* over every evaluation made and so an extreme-value
    statistic -- the luckiest draw, not the value the point returns now.

    The claim is made anyway, and the reason is worth stating because an
    earlier version of this gated it on ``pass_rng``. **Whether a
    loss is deterministic is not something a handshake may read.** A
    real problem does not come labelled, the caller usually cannot say,
    and an update rule that changes shape depending on how the objective
    was declared is not one you can reason about. So the behaviour is
    uniform.

    The cost of being uniform is small, and measured. The optimistic
    bias the claim introduces is the *same* bias the algorithm already
    has internally: MR15-GA's ``(µ+λ)`` selection keeps the better of
    each pair, so its own stored ``fitness`` is a running minimum too.
    Under noise its mutation scale collapses geometrically with **no
    switch at all** -- ``1 → 0.0039`` over six generations from a reset
    (``l2co ADR 0014``). Skipping the claim would not fix that and would
    cost something real: with own-ask the rebase is the only way the
    handed-over *location* reaches a mutation GA, so refusing it makes
    the handshake scale-only for them.

    What is still refused is a bundle whose numbers are unusable --
    a non-finite ``x`` or ``f``. That is a property of the values in
    hand, not of the problem, and ``+inf`` in particular is the opposite
    failure: every candidate beats it, so the 1/5 rule reads 100% and
    doubles the width immediately after the scale write set it.

    Parameters
    ----------
    repair_baseline : bool
        Whether this optimizer needs ``population`` / ``fitness``
        rebased on the bundle at all.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        updates: dict[str, Array] = {}
        use = _usable_sigma(bundle)

        std = getattr(opt_state, "std", None)
        if std is not None:
            std = jnp.asarray(std)
            if std.ndim == 0:
                updates["std"] = _blend(use, bundle.sigma, std)
            else:
                updates["std"] = _rescale_to(std, bundle.sigma, use)

        velocity = getattr(opt_state, "velocity", None)
        if velocity is not None:
            updates["velocity"] = _rescale_to(velocity, bundle.sigma, use)

        if repair_baseline:
            population = getattr(opt_state, "population", None)
            fitness = getattr(opt_state, "fitness", None)
            if population is not None and fitness is not None:
                # Both halves move together or neither does: rebasing
                # the points while keeping the old losses (or the
                # reverse) leaves the ``(mu + lambda)`` selection
                # incoherent, rows claiming a fitness they did not earn.
                #
                # ``f`` is gated as tightly as ``x``, and ``+inf`` is
                # the case that matters rather than NaN. ``best_loss``
                # starts at ``+inf`` and stays there while nothing has
                # improved -- a first generation that is entirely NaN,
                # which this ecosystem sees at high ambient dimension.
                # Writing that in makes *every* candidate beat the
                # stored elite, so the 1/5 rule reads a 100% success
                # rate and **doubles** the width immediately after the
                # scale write set it: measured success 1.00 against 0.00
                # for a finite baseline. That is precisely the failure
                # this repair exists to prevent, so an unusable ``f``
                # must leave the optimizer's own elite alone.
                repair_ok = _jumping(bundle) & jnp.isfinite(bundle.f)
                flat = jnp.concatenate(
                    [
                        jnp.ravel(jnp.asarray(leaf, dtype=float))
                        for leaf in jax.tree.leaves(bundle.x)
                    ]
                )
                updates["population"] = _blend(
                    repair_ok,
                    jnp.broadcast_to(flat, jnp.shape(population)),
                    jnp.asarray(population),
                )
                updates["fitness"] = _blend(
                    repair_ok,
                    jnp.full_like(jnp.asarray(fitness), bundle.f),
                    jnp.asarray(fitness),
                )

        return opt_state.replace(**updates) if updates else opt_state

    return write


def make_write_gradient(learning_rate: float) -> Callable:
    """Build the writer for an optax second-moment optimizer.

    The per-coordinate step is ``learning_rate * mu / (sqrt(nu) + eps)``,
    so scaling ``nu`` by ``k²`` scales the step by ``1/k``. To move the
    step RMS from its current value to ``sigma`` the second moment is
    multiplied by ``(rms_own / sigma)²``, which leaves the *relative*
    per-coordinate preconditioner untouched -- writing a flat value into
    ``nu`` would erase it.

    Parameters
    ----------
    learning_rate : float
        The rate the optimizer was constructed with.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        momentum, momentum_name = _get_first(opt_state, _MOMENTUM_NAMES)
        second, second_name = _get_first(opt_state, _SECOND_MOMENT_NAMES)
        if momentum is None and second is None:
            return opt_state
        if second is not None:
            second = jnp.asarray(second, dtype=float)

        if momentum is not None:
            momentum = jnp.asarray(momentum, dtype=float)
            step = momentum
            if second is not None:
                step = step / (jnp.sqrt(second) + 1e-8)
        else:
            # Preconditioner but no stored step (RMSProp, Adagrad,
            # Adafactor): the reference is the step the optimizer would
            # take for a unit gradient. That makes these optimizers
            # receivers even though the reader cannot make them
            # providers -- you cannot measure a step that was never
            # stored, but you can still rescale the preconditioner.
            step = 1.0 / (jnp.sqrt(second) + 1e-8)
        rms_own = float(learning_rate) * _rms(step)

        use = _usable_sigma(bundle)
        use = use & jnp.isfinite(rms_own) & (rms_own > 0.0)
        count, _ = _get_first(opt_state, _COUNT_NAMES)
        if count is not None:
            use = use & (jnp.asarray(count) > 0)

        ratio = jnp.where(
            use, rms_own / jnp.where(use, bundle.sigma, 1.0), 1.0
        )
        if second is not None:
            # Scaling the second moment by k² scales the step by 1/k,
            # leaving the relative per-coordinate preconditioner intact.
            return optax.tree.set(
                opt_state,
                **{
                    second_name: _blend(
                        use, second * jnp.square(ratio), second
                    )
                },
            )
        # Momentum-only (SGD with momentum, LARS): no preconditioner to
        # rescale, so the step scale lives in the trace itself.
        return optax.tree.set(
            opt_state,
            **{momentum_name: _blend(use, momentum / ratio, momentum)},
        )

    return write


def make_write_rprop(min_step_size: float, max_step_size: float) -> Callable:
    """Build the writer for Rprop.

    ``step_sizes`` is rescaled so its RMS matches ``sigma``, preserving
    the per-coordinate ratchet, then clipped back into the bounds Rprop
    was constructed with -- ``update_fn`` clips on every step, so a write
    that ignored them would be silently saturated on the next one.

    ``prev_updates`` is zeroed. It holds the previous signed step, and
    Rprop's ratchet keys on ``sign(g * prev_updates)``; across an iterate
    jump those signs describe a different neighbourhood, so keeping them
    fires a spurious increase or decrease on every coordinate whose
    gradient sign happened to flip. With ``prev_updates`` zero the
    product is zero, which is the branch that leaves ``step_sizes``
    untouched (``transform.py:928-940``).

    What that costs is **one zero step, then a clean one** -- not one
    clean step, as this said before. Optax emits the *previous* buffer
    rather than the step it just computed: ``updates = where(sign < 0,
    0, state.prev_updates)`` (``transform.py:949-953``, optax 0.2.8), so
    Rprop lags a generation always and its own first iteration moves
    nothing. Zeroing the buffer therefore spends one iteration -- and
    the feval that goes with it -- re-measuring the handed-over point
    without moving off it, before the ratchet's step lands on the next.
    The alternative is worse: any nonzero buffer we could invent is a
    step in a direction computed somewhere else, and it fires the
    ratchet on arrival. Fixing the lag itself belongs upstream; doing it
    locally would change Rprop's *static* trajectory, and with it the
    databank rows and the ERTD level that ``headroom4`` and
    ``bbob_headroom_{train,test}`` were jointly selected on.

    Parameters
    ----------
    min_step_size : float
        Lower clip bound the optimizer was constructed with.
    max_step_size : float
        Upper clip bound the optimizer was constructed with.

    Returns
    -------
    Callable
        ``writer(bundle, opt_state) -> opt_state``.
    """

    def write(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
        step_sizes = optax.tree.get(opt_state, "step_sizes")
        if step_sizes is None:
            return opt_state

        use = _usable_sigma(bundle)
        rescaled = _rescale_to(step_sizes, bundle.sigma, use)
        rescaled = jax.tree.map(
            lambda leaf: jnp.clip(leaf, min_step_size, max_step_size),
            rescaled,
        )

        opt_state = optax.tree.set(
            opt_state,
            step_sizes=jax.tree.map(
                lambda new, old: _blend(use, new, old), rescaled, step_sizes
            ),
        )
        prev = optax.tree.get(opt_state, "prev_updates")
        if prev is not None:
            jumping = _jumping(bundle)
            opt_state = optax.tree.set(
                opt_state,
                prev_updates=jax.tree.map(
                    lambda p: _blend(jumping, jnp.zeros_like(p), p), prev
                ),
            )
        return opt_state

    return write


def write_none(bundle: TransferBundle, opt_state: PyTree) -> PyTree:
    """Accept nothing, for optimizers with no slot a scale survives in.

    L-BFGS is the case: after ``init`` its state is all zeros with
    ``learning_rate = 1.0`` and ``value = inf``, and the first step
    length comes from ``identity_scale = min(1, 1/‖g‖)`` before the zoom
    line search overwrites it. There is nowhere for a scale to land.

    Parameters
    ----------
    bundle : TransferBundle
        Ignored.
    opt_state : PyTree
        Returned unchanged.

    Returns
    -------
    PyTree
        ``opt_state``.
    """
    del bundle
    return opt_state


#                                                              The policy
# =============================================================================


class TransferSpec(eqx.Module):
    """Which reader and writer one optimizer needs.

    Attributes
    ----------
    family : str
        One of :data:`FAMILIES`. Says what kind of search state the
        optimizer holds, and nothing more -- the population half of a
        switch is decided by :attr:`own_ask`, not by this.
    reader : str
        Key into the reader implementations.
    writer : str
        Key into the writer implementations.
    own_ask : bool
        Whether the incoming optimizer regenerates its own post-switch
        population through its ``ask``, instead of being handed a
        layout. True for every ``distribution`` algorithm and for the
        mutation GAs whose next ``ask`` or ``tell`` reads a baseline
        this handshake just wrote; see ``l2co ADR 0014``.
    """

    family: str = eqx.field(static=True)
    reader: str = eqx.field(static=True)
    writer: str = eqx.field(static=True)
    own_ask: bool = eqx.field(static=True, default=False)


_FAMILY_DEFAULTS: dict[str, TransferSpec] = {
    FAMILY_DISTRIBUTION: TransferSpec(
        FAMILY_DISTRIBUTION, "distribution", "distribution", own_ask=True
    ),
    FAMILY_POPULATION: TransferSpec(
        FAMILY_POPULATION, "population", "population"
    ),
    FAMILY_GRADIENT: TransferSpec(FAMILY_GRADIENT, "gradient", "gradient"),
}

TRANSFER_OVERRIDES: dict[str, TransferSpec] = {
    # Reads its own step length exactly; receives one exactly. The only
    # optimizer whose scale field is natively a length.
    "rprop": TransferSpec(FAMILY_GRADIENT, "rprop", "rprop"),
    # Its last accepted step is in the state; nothing survives a write.
    "lbfgs": TransferSpec(FAMILY_GRADIENT, "lbfgs", "none"),
    # The mutation GAs. Each adapts against its *own stored* baseline on
    # the next generation -- MR15-GA's 1/5 rule reads ``state.fitness``
    # (``mr15_ga.py:96``), GESMR-GA ranks its std groups by
    # ``fitness - state.fitness`` (``gesmr_ga.py`` ``_tell``, Eq. 5), and
    # SAMR-GA inherits per-member stds through ``argsort(state.fitness)``
    # -- so they need the repairing writer, and they draw their own
    # post-switch population from it. See ``l2co ADR 0014``.
    "mr15ga": TransferSpec(
        FAMILY_POPULATION, "population", "population_repair", own_ask=True
    ),
    "samrga": TransferSpec(
        FAMILY_POPULATION, "population", "population_repair", own_ask=True
    ),
    "gesmrga": TransferSpec(
        FAMILY_POPULATION, "population", "population_repair", own_ask=True
    ),
    # Holds no persistent search state at all.
    "randomsearch": TransferSpec(FAMILY_POPULATION, "none", "none"),
}
"""Per-optimizer departures from the family default, keyed as in the registry.

Deliberately short. The family default is right for the large majority,
and every entry here exists because a specific field was read and found
to behave differently -- not because the optimizer is unusual in general.

``simple_ga`` is deliberately **absent** even though it is a mutation GA
of the same shape: its ``_tell`` overwrites ``std`` from a fixed
schedule (``simple_ga.py`` ``_tell``), so a transferred scale expires on
the next generation, and its ``_tell`` replaces the population wholesale
rather than selecting ``(mu + lambda)``, so there is no stored baseline
to repair. It stays a scale *provider* on the family default.
"""


def transfer_spec_for(name: str, family: str) -> TransferSpec:
    """Resolve the transfer spec for one optimizer.

    Parameters
    ----------
    name : str
        Registry name, normalised as in
        :func:`~l2co_optimizers._src.core.utils.normalize_key`.
    family : str
        One of :data:`FAMILIES`, supplied by the constructing factory,
        which knows which family it is building. Not inferred from
        registry membership: ``FEEDBACK_evosax.md`` records that
        membership-based family dispatch is already a source of friction.

    Returns
    -------
    TransferSpec
        The override for ``name`` if one exists, else the family default.

    Raises
    ------
    ValueError
        If ``family`` is not one of :data:`FAMILIES`, or is
        :data:`FAMILY_DERIVATIVE_FREE`, a plain-run family with no
        transfer spec.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}; got {family!r}")
    if family not in _FAMILY_DEFAULTS:
        raise ValueError(
            f"family {family!r} is held only by plain-run entries, which "
            f"never take part in a switch, so it has no transfer spec "
            f"(ADR 0002); got optimizer {name!r}"
        )
    return TRANSFER_OVERRIDES.get(name, _FAMILY_DEFAULTS[family])


def build_transfer_fns(
    name: str,
    family: str,
    hyperparameters: dict | None = None,
    ravel_fn: Callable | None = None,
) -> tuple[TransferReadFunction, TransferWriteFunction]:
    """Bind one optimizer's reader and writer at construction time.

    Both halves need values that are not recoverable from the state, so
    they are closed over here rather than looked up per call: the
    learning rate (``scale_by_learning_rate`` holds an ``EmptyState``),
    Rprop's clip bounds, and the ravel function evosax uses to flatten a
    solution pytree.

    Parameters
    ----------
    name : str
        Registry name.
    family : str
        One of :data:`FAMILIES`.
    hyperparameters : dict or None, optional
        The hyperparameters the optimizer was constructed with. Missing
        keys fall back to the underlying library's documented defaults.
    ravel_fn : Callable or None, optional
        The algorithm's ``_ravel_solution``. Required for
        ``distribution`` -- evosax stores ``mean`` raveled.

    Returns
    -------
    tuple[TransferReadFunction, TransferWriteFunction]
        ``(reader, writer)``. ``reader(opt_state) -> (sigma, conf)``;
        ``writer(bundle, opt_state) -> opt_state``.

    Raises
    ------
    ValueError
        If a ``distribution`` writer is requested without ``ravel_fn``.
    """
    hyperparameters = dict(hyperparameters or {})
    spec = transfer_spec_for(name, family)

    learning_rate = hyperparameters.get("learning_rate", 1.0)

    readers: dict[str, Callable] = {
        "distribution": read_sigma_distribution,
        "population": read_sigma_population,
        "gradient": make_read_sigma_gradient(learning_rate),
        "rprop": read_sigma_rprop,
        "lbfgs": make_read_sigma_lbfgs(
            hyperparameters.get("scale_init_precond", True),
            hyperparameters.get("learning_rate", None),
        ),
        "none": read_sigma_absent,
    }

    if spec.writer == "distribution" and ravel_fn is None:
        raise ValueError(
            "a 'distribution' writer needs ravel_fn: evosax stores "
            "state.mean raveled, so the bundle's pytree x must be "
            "flattened with the algorithm's own _ravel_solution"
        )

    writers: dict[str, Callable] = {
        "distribution": (
            make_write_distribution(ravel_fn)
            if ravel_fn is not None
            else write_none
        ),
        "population": make_write_population(repair_baseline=False),
        "population_repair": make_write_population(repair_baseline=True),
        "gradient": make_write_gradient(learning_rate),
        "rprop": make_write_rprop(
            hyperparameters.get("min_step_size", 1e-6),
            hyperparameters.get("max_step_size", 50.0),
        ),
        "none": write_none,
    }

    return readers[spec.reader], writers[spec.writer]


# =============================================================================
