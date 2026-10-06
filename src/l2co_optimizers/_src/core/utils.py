"""Utility helpers shared across optimizer implementations."""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import re

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# TODO: This should normalize_key should live in the mapping.py module
# , not here


def normalize_key(key: str) -> str:
    """Strip non-alphanumerics and lowercase ``key`` for mapping lookups."""
    return re.sub(r"[^a-zA-Z0-9]", "", key).lower()
