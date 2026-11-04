"""How far an optimizer can enforce a task's constraints (ADR 0008).

Every registry entry declares one of three levels. The levels describe
the optimizer, not the run: what its algorithm can enforce, whether or
not a run hands it a constraint. Today a run can hand it only a box,
the factories' ``bounded`` pair (l2co ADR 0020).
"""

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

CONSTRAINTS_NONE = "none"
"""The algorithm never learns about a constraint.

A ``bounded`` pair is still applied, by clipping each candidate before
it is evaluated or after each step. The optimizer's own state (Adam's
moments, L-BFGS's memory, a search distribution, a simplex) is left as
the unclipped step made it, so the next step does not know that the
point it came from was moved.
"""

CONSTRAINTS_BOX = "box"
"""The optimizer keeps its own state inside a box.

Its solver takes the bounds itself (L-BFGS-B, TNC, Powell), it repairs
or clips candidates before its state learns from them (SHADE, TuRBO,
the RBF trust region), or it has no state to keep (random search).
"""

CONSTRAINTS_GENERAL = "general"
"""The solver also takes inequality and equality constraints.

Implies :data:`CONSTRAINTS_BOX`. Nothing passes an inequality or an
equality to an optimizer yet: l2co refuses a task that has one (l2co
ADR 0020).
"""

CONSTRAINT_HANDLING_LEVELS = (
    CONSTRAINTS_NONE,
    CONSTRAINTS_BOX,
    CONSTRAINTS_GENERAL,
)
"""The levels, weakest first: a later level covers every earlier one."""
