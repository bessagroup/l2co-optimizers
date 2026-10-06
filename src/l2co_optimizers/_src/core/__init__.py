"""The optimizer-agnostic core of ``l2co_optimizers``.

What an optimizer *is* and how it runs, independent of any particular
optimizer: the contract types, ``UpdateClass`` and its run loop, the
states it threads, the ``OptimizationStep`` spec, loss helpers,
samplers, popsize rules, stopping criteria, and the state-transfer
ports every optimizer plugs into.

Nothing here imports a specific optimizer: the implementations (optax,
evosax, L-BFGS, SHADE, TuRBO, RBF trust region, random search) and the
modules that must see all of them (the registry in ``mapping``, and
``optimizer_parts``) live one level up, in ``_src``. The handshake
policy and the rest of the switching layer live in l2co (l2co ADR
0019).
``tests/test_core_layering.py`` enforces the direction.
"""

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
