"""``random_sampling`` draws uniformly from ``[lower_bound, upper_bound)``.

It used to draw from ``[lower_bound, lower_bound + upper_bound)``: the
initializer's scale was ``upper_bound`` where it should be the width
``upper_bound - lower_bound``. The two agree only when ``lower_bound`` is
0, so the default ``(0, 1)`` draws must stay exactly what they were.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import jax
import jax.numpy as jnp
import pytest
from jax.nn import initializers

# Local
from l2co_optimizers import random_sampling
from l2co_optimizers._src.core.sampler import _sample

# =============================================================================

_PARAMS = {"w": jnp.zeros((3,)), "b": jnp.zeros(())}
_N = 2000


@pytest.mark.parametrize(
    "lower, upper",
    [(-5.0, 5.0), (2.0, 3.0), (-1.0, -0.5)],
    ids=["straddles-zero", "positive", "negative"],
)
def test_samples_cover_exactly_the_requested_range(lower, upper):
    samples = random_sampling(
        jax.random.key(0), _PARAMS, _N, lower_bound=lower, upper_bound=upper
    )
    values = jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree.leaves(samples)]
    )
    assert bool(jnp.all(values >= lower))
    assert bool(jnp.all(values < upper))
    # Spread over the whole range, not a sub-interval of it.
    width = upper - lower
    assert float(values.min()) < lower + 0.05 * width
    assert float(values.max()) > upper - 0.05 * width


def test_default_bounds_draw_what_they_always_did():
    key = jax.random.key(1)
    before = _sample(initializers.uniform(scale=1.0), _N, key, _PARAMS)
    for got in (
        random_sampling(key, _PARAMS, _N),
        random_sampling(key, _PARAMS, _N, lower_bound=0.0, upper_bound=1.0),
    ):
        for a, b in zip(
            jax.tree.leaves(got), jax.tree.leaves(before), strict=True
        ):
            assert bool(jnp.array_equal(a, b))
