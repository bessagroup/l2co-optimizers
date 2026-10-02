"""Which handshake an optimizer needs when control passes to it.

A *handshake* is the state transfer applied when the active optimizer
changes: which population the incoming optimizer sees, and whether its
internal state is reset or continued. Which variant a given optimizer
needs is a property of the **optimizer** -- can its accumulated internal
state survive an iterate jump? -- not of the loop doing the switching.
This module holds that table, and only that table.

The **population** half's transfer functions live here too, and are the
single canonical implementation. They used to be duplicated per consumer
with divergent slot layouts -- one placing the best point in the last
slot and drawing ``popsize`` samples, the other in slot 0 drawing
``popsize - 1``. That is not cosmetic. CR-FM-NES draws
``concat([z, -z])`` (``cr_fm_nes.py:176-177``), so its antithetic
partners sit at ``(i, i + popsize / 2)``, and -- more to the point -- it
*caches* those draws in state and consumes them on the next ``tell``,
pairing ``fitness`` with ``state.z`` index for index (``:201``,
``:207``, ``:235``). Which slot the best point occupies therefore
decides which cached search direction its fitness is credited to, and
the two layouts produce *different* updates from the same handshake. A
policy trained against one layout and deployed against the other
diverges at every switch. (An earlier version of this note attributed
the divergence to adjacent pairs at ``(2i, 2i + 1)`` and to population
geometry; the conclusion held but the mechanism did not.) The deeper
problem -- that substituting *any* foreign population invalidates those
cached draws -- is what ``l2co ADR 0012`` addresses, by regenerating the
population through the incoming optimizer's own ``ask`` instead.

The canonical layout is the environment's (best point
last), chosen because it is what existing trained checkpoints and stored
trajectories were produced with. This amends
``l2co ADR 0009-shared-handshake-policy-table.md``, which recorded the
divergence as accepted.

The **opt_state** half's transfer functions are still per-consumer,
because they genuinely operate on incompatible data models: the rl2co
environment's on ``RunState`` / ``UpdateClass``, the deployment
wrappers' on :class:`~l2co_optimizers._src.sub_optimizer.SubOpt` and
their own strategy ``State``. Only the *decision* is shared there.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from l2co_optimizers._src.optimizer_schedule import OptimizationStep
from l2co_optimizers._src.typing import SamplerFunction

# =============================================================================

__all__ = [
    "DEFAULT_HANDSHAKE_POLICY",
    "HANDSHAKE_POLICY",
    "POPULATION_HANDSHAKES",
    "population_best",
    "population_best_one",
    "population_handshake_for",
    "OPT_STATE_VARIANTS",
    "POPULATION_VARIANTS",
    "HandshakePolicy",
    "handshake_policy_for",
]


class HandshakePolicy(eqx.Module):
    """Which handshake variants one optimizer needs on a switch.

    Two orthogonal halves. Both fields are static: a
    ``HandshakePolicy`` contributes no array leaves to a surrounding
    pytree.

    Attributes
    ----------
    population : str
        ``"best"`` -- the incoming optimizer sees the best parameters
        so far, expanded to its (unit) population. ``"best_one"`` -- it
        sees a freshly sampled population with the best injected.
    opt_state : str
        ``"continue"`` -- the optimizer's internal state carries over,
        so switching back resumes its own adaptation. ``"reset"`` -- the
        internal state is re-initialised, for optimizers whose
        accumulated state cannot survive an iterate jump. "Re-initialised
        *around the best parameters*" would overstate it: L-BFGS, the one
        tabulated ``"reset"``, has an ``init`` that reads its argument's
        shape and dtype only (``optax`` ``transform.py:1719`` stores
        ``zeros_like(params)``), so the values passed do not reach the
        state. An optimizer whose ``init`` *does* read them -- any evosax
        distribution-based algorithm, which takes its ``mean`` from the
        argument -- would make the point handed in load-bearing, and the
        caller must then pass the best parameters rather than whatever
        population it happens to hold.
    """

    population: str = eqx.field(static=True)
    opt_state: str = eqx.field(static=True)


POPULATION_VARIANTS = ("best", "best_one")
OPT_STATE_VARIANTS = ("reset", "continue")

DEFAULT_HANDSHAKE_POLICY = HandshakePolicy(
    population="best_one", opt_state="continue"
)
"""Fallback for optimizers absent from :data:`HANDSHAKE_POLICY`.

The optimizer registry is extensible (``register_optimizer``), so any
consumer may be handed an optimizer this table has no entry for. Rather
than raising, those fall back to ``best_one`` + ``continue``.

This pair is the conservative choice: ``best_one`` degenerates to
``best`` whenever ``popsize == 1`` (the default for gradient-based
optimizers), so the fallback matches the tabulated behaviour for both
gradient-based and population-based entries. It differs only for
optimizers that need their internal state *reset* on a switch --
``lbfgs`` is the one known case, and it is tabulated explicitly. Add an
entry for any new optimizer whose internal state cannot survive an
iterate jump.
"""

HANDSHAKE_POLICY: dict[str, HandshakePolicy] = {
    "adam": HandshakePolicy("best", "continue"),
    "adam_variable_updates": HandshakePolicy("best", "continue"),
    "sgd": HandshakePolicy("best", "continue"),
    "sepcmaes": HandshakePolicy("best_one", "continue"),
    "cmaes": HandshakePolicy("best_one", "continue"),
    "pso": HandshakePolicy("best_one", "continue"),
    "shade": HandshakePolicy("best_one", "continue"),
    # L-BFGS carries the best candidate but must wipe its curvature
    # memory and line-search state: both describe the neighbourhood of
    # the iterate it was building them at, and the handshake jumps the
    # iterate somewhere else.
    "lbfgs": HandshakePolicy("best", "reset"),
}
"""Handshake policy per optimizer name, keyed as in the registry."""


def handshake_policy_for(
    optimizer: OptimizationStep | str,
) -> HandshakePolicy:
    """Resolve the handshake policy for an optimizer.

    Parameters
    ----------
    optimizer : OptimizationStep | str
        Optimizer spec, or its registry name directly.

    Returns
    -------
    HandshakePolicy
        The tabulated policy, or :data:`DEFAULT_HANDSHAKE_POLICY` for
        names absent from :data:`HANDSHAKE_POLICY`.
    """
    name = (
        optimizer.optimizer
        if isinstance(optimizer, OptimizationStep)
        else optimizer
    )
    return HANDSHAKE_POLICY.get(name, DEFAULT_HANDSHAKE_POLICY)


# =============================================================================


#                                          Population handshake implementations
# =============================================================================


def population_best(
    best_params: PyTree,
    popsize: int,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> PyTree:
    """Hand the incoming optimizer the best parameters in every slot.

    The ``"best"`` variant. For the unit populations this variant is
    tabulated for (``adam``, ``sgd``, ``lbfgs``) that is a single row,
    identical to what a ``"best_one"`` handshake would produce -- the two
    only diverge above ``popsize == 1``.

    Parameters
    ----------
    best_params : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    popsize : int
        Population size of the incoming optimizer.
    sampler : SamplerFunction
        Unused; accepted so both variants share one signature.
    key : PRNGKeyArray
        Unused; accepted so both variants share one signature.

    Returns
    -------
    PyTree
        ``best_params`` with a leading axis of ``popsize``.
    """
    del sampler, key
    return jax.tree.map(
        lambda b: jnp.broadcast_to(b, (popsize, *jnp.shape(b))), best_params
    )


def population_best_one(
    best_params: PyTree,
    popsize: int,
    sampler: SamplerFunction,
    key: PRNGKeyArray,
) -> PyTree:
    """Hand over a fresh population with the best parameters injected.

    The ``"best_one"`` variant: ``popsize`` draws from ``sampler``, with
    ``best_params`` written into the **last** slot. Note the draws are
    *not* centred on ``best_params`` -- every sampler in
    :mod:`l2co_optimizers._src.sampler` uses ``params`` for shape and dtype
    only (``_sample``), so this is a fresh draw from the sampler's own
    distribution with the best point injected, not a local perturbation
    of it.

    That slot is load-bearing, which is why this implementation is shared
    rather than reproduced per consumer: CR-FM-NES caches its own draws
    in state and credits each candidate's fitness to the cached
    direction at the same index, so moving the best point yields a
    different update from the same handshake. Last is canonical because
    it is what existing trained checkpoints and stored trajectories were
    produced with.

    Parameters
    ----------
    best_params : PyTree
        Best parameters seen so far, shaped as one solution (no leading
        population axis).
    popsize : int
        Population size of the incoming optimizer; also the number of
        draws requested, one of which the injected best point replaces.
    sampler : SamplerFunction
        ``l2co.sampling`` function, called as
        ``sampler(key, best_params, popsize)``.
    key : PRNGKeyArray
        Key for the draws.

    Returns
    -------
    PyTree
        Population with a leading axis of ``popsize``.
    """
    samples = sampler(key, best_params, popsize)
    return jax.tree.map(lambda s, b: s.at[-1].set(b), samples, best_params)


POPULATION_HANDSHAKES: dict[str, Callable[..., PyTree]] = {
    "best": population_best,
    "best_one": population_best_one,
}
"""The ``population`` half's implementations, keyed as in the table.

Keys match :data:`POPULATION_VARIANTS`. All share the signature
``(best_params, popsize, sampler, key) -> PyTree``, returning a
population with a leading axis of ``popsize``. A consumer whose
generation width exceeds ``popsize`` (the strategy wrappers, which
allocate ``max(popsize)`` across the menu) pads the result itself --
padding is the caller's layout concern, not the handshake's.
"""


def population_handshake_for(
    optimizer: OptimizationStep | str,
) -> Callable[..., PyTree]:
    """Resolve the population-handshake implementation for an optimizer.

    Parameters
    ----------
    optimizer : OptimizationStep | str
        Optimizer spec, or its registry name directly.

    Returns
    -------
    Callable[..., PyTree]
        The tabulated variant's implementation, or the default's for
        names absent from :data:`HANDSHAKE_POLICY`.
    """
    return POPULATION_HANDSHAKES[handshake_policy_for(optimizer).population]
