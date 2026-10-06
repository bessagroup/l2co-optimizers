"""Menu-dispatched population evaluation.

A strategy wrapper emits every population at one fixed width -- the
**generation width**, ``max(popsize)`` over the menu -- because a
single ``ask`` must return a single shape whichever optimizer is
active. Only the active optimizer's own **evaluation width** carries
real candidates, though: ``popsize`` rows for a population-based
optimizer, one row (the iterate) for a gradient-based one. The
remainder are **padding rows**, carried as ``NaN``.

:func:`menu_loss_and_grad` evaluates only the evaluation width. It
builds one :func:`jax.lax.switch` branch per *distinct* ``(kind,
width)`` pair in the menu and dispatches on the active menu index, so
a gradient generation never pays the population-wide forward pass and
a population generation never pays a backward pass it would discard.
See ``l2co ADR 0011-per-optimizer-evaluation-width-on-the-strategy-
bridge.md``.

**Branches are deduplicated, but not for speed.** Under
:func:`jax.vmap` -- which is how the batched driver
(:meth:`UpdateClass.batch_run_fused`) runs ``n_realizations``
trajectories at once -- the dispatch index is per-realization, so
``lax.switch`` lowers to a ``select`` over *every* branch and all of
them execute. That makes it look as though collapsing the canonical
four-entry menu ``[sepcmaes, adam, lbfgs, pso]`` to its two distinct
``(kind, width)`` pairs should halve the work. Measured, it does not:
XLA's CSE already collapses identical branch bodies, so four branches
cost the same as two. Deduplication survives only because it cuts
trace-and-compile time (~1.35x on that menu) and because
:attr:`branches` is then an honest count of a menu's distinct
evaluation shapes.

**Gradient generations evaluate exactly one row.** That is the
iterate, ``population[0]`` -- the only row any gradient sub-optimizer
consumes. It also means the batched pass runs at width one, where XLA
takes a degenerate-vmap code path whose gradient differs from the wide
pass by 4.33e-06 relative. Widths of two and above are bit-identical to
the wide pass; width one is not. Chosen knowingly: on the canonical
menu a gradient generation costs 41.7 ms at width 1 against 131.7 ms at
width 2 (26.7x vs 8.5x the 1114 ms pre-dispatch pass), so results are
semantically identical to but not bit-reproducible against runs
predating it.

**No menu bit-reproduces pre-dispatch runs.** Gradient menus shift for
the reason above. Population menus shift too, on some platforms --
*not* because the backward pass is gone, but because dispatching emits
a different HLO graph than a bare full-width call (a slice, which is a
no-op on the values but not on the graph, plus a conditional) and XLA
may fuse it differently. Identical on x86_64 Linux; up to one float32
ulp apart on macOS/arm64. Bit-exactness here is not an assertable
property -- see ``l2co ADR 0011``.

**NaN contract.** Returned ``loss`` and ``grads`` are ``NaN`` in a row
whenever that row is padding *or* its parameters are non-finite. The
second half matters because a ``loss_fn`` that clips or masks can
return a finite value from ``NaN`` inputs, and best-tracking
(``_update_best``) and the NaN-aware history reductions
(``HistoryState.from_history``) both rely on a diverged candidate
being unselectable.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.loss import (
    vmapped_loss,
    vmapped_loss_and_grad,
    vmapped_loss_and_grad_with_rng,
    vmapped_loss_with_rng,
)
from l2co_optimizers._src.core.typing import LossFunction

if TYPE_CHECKING:
    from l2co_optimizers._src.sub_optimizer import SubOpt

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = ["GRAD_EVALUATION_WIDTH", "menu_loss_and_grad"]


#: Rows a gradient-based sub-optimizer's generation evaluates. Only
#: ``population[0]`` (the iterate) is ever consumed by a gradient step,
#: so this is ``1``. See the module docstring on why widening it to
#: ``2`` would restore bit-exactness with the pre-dispatch path.
GRAD_EVALUATION_WIDTH = 1


#                                                                     Internals
# =============================================================================


def row_has_nan(population: PyTree) -> Array:
    """Flag rows of ``population`` containing any ``NaN``.

    Parameters
    ----------
    population : PyTree
        Population pytree whose every leaf has leading axis
        ``population_size``.

    Returns
    -------
    Array
        Boolean array of shape ``(population_size,)``, ``True`` where
        that row has at least one ``NaN`` in any leaf.
    """
    leaves = jax.tree.leaves(population)
    mask = jnp.zeros((leaves[0].shape[0],), dtype=bool)
    for leaf in leaves:
        mask = mask | jnp.isnan(leaf).any(axis=tuple(range(1, leaf.ndim)))
    return mask


def _pad_nan_axis0(arr: Array, target: int) -> Array:
    """Extend ``arr`` along axis 0 to ``target`` rows with ``NaN``.

    Parameters
    ----------
    arr : Array
        Array whose leading axis is at most ``target``.
    target : int
        Desired leading-axis length.

    Returns
    -------
    Array
        ``arr`` unchanged when already ``target`` rows long, else
        ``arr`` followed by ``NaN`` rows. Dtype is preserved.
    """
    pad = target - arr.shape[0]
    if pad == 0:
        return arr
    tail = jnp.full((pad, *arr.shape[1:]), jnp.nan, dtype=arr.dtype)
    return jnp.concatenate([arr, tail], axis=0)


def _branch_table(
    sub_optimizers: tuple[SubOpt, ...],
) -> tuple[tuple[tuple[str, int], ...], tuple[int, ...]]:
    """Collapse a menu to its distinct evaluation branches.

    Parameters
    ----------
    sub_optimizers : tuple[SubOpt, ...]
        The menu adapters, in menu (logit) order.

    Returns
    -------
    tuple[tuple[tuple[str, int], ...], tuple[int, ...]]
        ``(branches, menu_to_branch)`` where ``branches`` lists the
        distinct ``("pop" | "grad", width)`` pairs in first-appearance
        order and ``menu_to_branch[i]`` is the branch index serving
        menu entry ``i``.
    """
    branches: list[tuple[str, int]] = []
    menu_to_branch: list[int] = []
    for sub in sub_optimizers:
        if sub.is_pop:
            spec = ("pop", int(sub.popsize))
        else:
            spec = ("grad", GRAD_EVALUATION_WIDTH)
        if spec not in branches:
            branches.append(spec)
        menu_to_branch.append(branches.index(spec))
    return tuple(branches), tuple(menu_to_branch)


#                                                                       Factory
# =============================================================================


def menu_loss_and_grad(
    sub_optimizers: tuple[SubOpt, ...],
    population_size: int,
    static: PyTree,
    loss_fn: LossFunction,
    pass_rng: bool,
) -> Callable[
    [Int[Array, ""], PyTree, dict, PRNGKeyArray],
    tuple[Float[Array, " population_size"], PyTree],
]:
    """Build a menu-dispatched population evaluator.

    The returned callable evaluates only the active optimizer's
    evaluation width (see the module docstring) and returns
    generation-width results, ``NaN``-filled beyond it.

    Parameters
    ----------
    sub_optimizers : tuple[SubOpt, ...]
        The menu adapters in menu (logit) order. Only the static
        ``is_pop`` and ``popsize`` fields are read, so any object
        exposing those works.
    population_size : int
        The generation width -- the leading axis of every population
        the caller will pass in. Must be at least the widest
        evaluation width in the menu.
    static : PyTree
        Non-inexact-array partition of the model, recombined with
        each candidate row via :func:`equinox.combine`.
    loss_fn : LossFunction
        The loss being optimized.
    pass_rng : bool
        Whether ``loss_fn`` takes a ``key`` keyword.

    Returns
    -------
    Callable
        ``eval_fn(active_opt_idx, population, sample, eval_keys) ->
        (loss, grads)``. ``loss`` has shape ``(population_size,)``;
        ``grads`` matches ``population``'s structure and shape. Both
        are ``NaN`` in padding rows and in rows whose parameters are
        non-finite. Gradients are all-``NaN`` on a population
        generation, matching what l2co's native population factories
        record in ``OptHistory``.

        The callable carries two introspection attributes:
        ``branches``, the distinct ``("pop" | "grad", width)`` pairs
        actually compiled, and ``menu_to_branch``, mapping each menu
        index to its branch index.

    Raises
    ------
    ValueError
        If ``sub_optimizers`` is empty, or ``population_size`` is
        smaller than the widest evaluation width in the menu.

    Notes
    -----
    ``eval_keys`` must be ``population_size`` keys long (as produced
    by ``jax.random.split(key, population_size)``). Row ``j`` is
    evaluated under ``eval_keys[j]`` exactly as an undispatched
    full-width evaluation would, so narrowing the width does not
    perturb which key any surviving row sees.
    """
    if not sub_optimizers:
        raise ValueError("sub_optimizers must be non-empty")

    branches, menu_to_branch = _branch_table(sub_optimizers)
    widest = max(width for _kind, width in branches)
    if population_size < widest:
        raise ValueError(
            f"population_size={population_size} is below the widest "
            f"evaluation width in the menu ({widest}); the generation "
            f"width must be at least max(popsize)."
        )

    branch_of_menu = jnp.asarray(menu_to_branch, dtype=int)

    def make_branch(spec: tuple[str, int]) -> Callable:
        """Build the evaluation branch for one ``(kind, width)`` pair."""
        kind, width = spec

        if kind == "pop":

            def pop_branch(
                population: PyTree,
                sample: dict,
                eval_keys: PRNGKeyArray,
            ) -> tuple[Array, PyTree]:
                """Value-only over ``width`` rows; grads all-``NaN``."""
                rows = jax.tree.map(lambda p: p[:width], population)
                model = eqx.combine(rows, static)
                if pass_rng:
                    loss = vmapped_loss_with_rng(
                        model, loss_fn, sample, eval_keys[:width]
                    )
                else:
                    loss = vmapped_loss(model, loss_fn, sample)
                grads = jax.tree.map(
                    lambda p: jnp.full_like(p, jnp.nan), population
                )
                return _pad_nan_axis0(loss, population_size), grads

            return pop_branch

        def grad_branch(
            population: PyTree,
            sample: dict,
            eval_keys: PRNGKeyArray,
        ) -> tuple[Array, PyTree]:
            """Value+grad over the iterate only; tail rows ``NaN``."""
            rows = jax.tree.map(lambda p: p[:width], population)
            model = eqx.combine(rows, static)
            if pass_rng:
                loss, grads = vmapped_loss_and_grad_with_rng(
                    model, loss_fn, sample, eval_keys[:width]
                )
            else:
                loss, grads = vmapped_loss_and_grad(model, loss_fn, sample)
            grads = jax.tree.map(
                lambda g, p: jnp.concatenate(
                    [
                        g,
                        jnp.full(
                            (population_size - width, *p.shape[1:]),
                            jnp.nan,
                            dtype=g.dtype,
                        ),
                    ],
                    axis=0,
                ),
                grads,
                population,
            )
            return _pad_nan_axis0(loss, population_size), grads

        return grad_branch

    branch_fns = tuple(make_branch(spec) for spec in branches)

    def eval_fn(
        active_opt_idx: Int[Array, ""],
        population: PyTree,
        sample: dict,
        eval_keys: PRNGKeyArray,
    ) -> tuple[Float[Array, " population_size"], PyTree]:
        """Evaluate one generation at the active optimizer's width.

        Parameters
        ----------
        active_opt_idx : Int[Array, ""]
            Menu index of the optimizer producing this generation.
        population : PyTree
            Candidates, leading axis ``population_size``.
        sample : dict
            Current data batch, forwarded to ``loss_fn``.
        eval_keys : PRNGKeyArray
            ``population_size`` per-row keys; consulted only when
            ``pass_rng``.

        Returns
        -------
        tuple[Float[Array, " population_size"], PyTree]
            ``(loss, grads)`` -- see :func:`menu_loss_and_grad`.
        """
        # A single branch when the menu needs only one: skip the
        # switch so the traced graph is what a hand-written
        # single-optimizer evaluation would be.
        if len(branch_fns) == 1:
            loss, grads = branch_fns[0](population, sample, eval_keys)
        else:
            loss, grads = jax.lax.switch(
                branch_of_menu[active_opt_idx],
                branch_fns,
                population,
                sample,
                eval_keys,
            )

        # Fold the diverged-candidate mask in: padding rows are already
        # NaN by construction, but a ``loss_fn`` that clips or masks can
        # return a finite value from NaN parameters, and best-tracking
        # must never adopt such a row.
        is_bad = row_has_nan(population)
        loss = jnp.where(is_bad, jnp.nan, loss)
        grads = jax.tree.map(
            lambda g: jnp.where(
                is_bad.reshape((is_bad.shape[0],) + (1,) * (g.ndim - 1)),
                jnp.nan,
                g,
            ),
            grads,
        )
        return loss, grads

    eval_fn.branches = branches
    eval_fn.menu_to_branch = menu_to_branch
    return eval_fn
