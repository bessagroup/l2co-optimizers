"""
Module for samplers.
"""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable

# Third-party
import jax
import jax.numpy as jnp
from jax.nn import initializers
from jaxtyping import PRNGKeyArray, PyTree

# Local
from l2co_optimizers._src.core.typing import InputParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def _sample(
    init_fn: Callable[[PRNGKeyArray, tuple[int], type], PyTree],
    n_samples: int,
    key: PRNGKeyArray,
    params: PyTree,
) -> InputParameters:
    """Generate samples using a JAX initializer function.

    Parameters
    ----------
    init_fn : Callable[[PRNGKeyArray, tuple[int], type], PyTree]
        JAX initializer function that generates samples.
    n_samples : int
        Number of samples to generate.
    key : PRNGKeyArray
        Random key for reproducibility.
    params : PyTree
        Parameters to sample (used for shape reference).

    Returns
    -------
    InputParameters
        Sampled parameters with shape ``(n_samples, ...)``.
    """
    keys = jax.random.split(key, len(jax.tree.leaves(params)))

    def init_leaf(x: jax.Array, k: PRNGKeyArray) -> jax.Array:
        """Sample a single leaf using the initializer."""
        return init_fn(k, (n_samples,) + x.shape, x.dtype)

    return jax.tree.map(
        init_leaf, params, jax.tree.unflatten(jax.tree.structure(params), keys)
    )


# =============================================================================
def random_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> InputParameters:
    """
    Sample uniformly from ``[lower_bound, upper_bound)``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample.
    n_samples : int
        Number of samples to generate.
    lower_bound : float, optional
        Lower end of the range (inclusive), by default 0.0.
    upper_bound : float, optional
        Upper end of the range (exclusive), by default 1.0.

    Returns
    -------
    InputParameters
        Randomly sampled parameters.
    """
    # The initializer draws from [0, scale), so the scale is the width of
    # the range. It used to be ``upper_bound``, which drew from
    # [lower_bound, lower_bound + upper_bound): the two agree only when
    # ``lower_bound`` is 0, so the default draws are unchanged.
    random_samples = _sample(
        init_fn=initializers.uniform(scale=upper_bound - lower_bound),
        n_samples=n_samples,
        key=key,
        params=params,
    )
    # Fold ``lower_bound`` in as a scalar broadcast instead of building a
    # full ``constant_sampling`` array and adding it: that materialised a
    # second sample-sized buffer (all zeros in the common ``lower_bound=0``
    # case) for no reason. A scalar add broadcasts and stays fusible; a
    # zero offset is skipped entirely.
    if lower_bound == 0.0:
        return random_samples
    return jax.tree.map(lambda x: x + lower_bound, random_samples)


def normal_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> PyTree:
    """
    Perform normal (Gaussian) sampling for a given PyTree of parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample (used for shape reference).
    n_samples : int
        Number of samples to generate.
    mean : float, optional
        Mean of the normal distribution, by default 0.0.
    std : float, optional
        Standard deviation of the normal distribution, by default 1.0.

    Returns
    -------
    PyTree
        Randomly sampled parameters from a normal distribution.
    """
    normal_samples = _sample(
        init_fn=initializers.normal(stddev=std),
        n_samples=n_samples,
        key=key,
        params=params,
    )

    constant_samples = constant_sampling(
        key, params=params, value=mean, n_samples=n_samples
    )

    return jax.tree.map(lambda x, y: x + y, normal_samples, constant_samples)


def relative_normal_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    std: float = 1.0,
) -> InputParameters:
    """
    Sample around ``params``, spread relative to each value's size.

    The one sampler that reads the *values* of ``params`` rather than only
    their shape (ADR 0004). A task that prescribes a starting point keeps
    it as its model -- a CUTEst task's model is the problem's ``x0`` --
    and this sampler is how its runs start there:

    * **member 0 is exactly** ``params``, so every optimizer with a
      population of one, and every distribution strategy's mean, starts
      at the prescribed point;
    * **member** ``k >= 1`` **is** ``x + std * max(|x|, 1) * N(0, 1)``,
      drawn independently per coordinate. The spread follows each
      coordinate's size, since CUTEst variables differ in scale by orders
      of magnitude, and falls back to ``std`` where ``|x| < 1``, so a zero
      component still spreads. DE and SHADE move along differences
      between members, so they need that spread.

    Called after a switch with the incumbent (l2co's handshake), it puts
    the incumbent in twice and draws the rest around it rather than
    around the origin; ADR 0004 accepts both.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample around; their values are member 0.
    n_samples : int
        Number of samples to generate.
    std : float, optional
        Spread relative to ``max(|x|, 1)``, by default 1.0. With ``1.0``
        and coordinates no larger than 1 in magnitude, the spread matches
        :func:`normal_sampling`'s default.

    Returns
    -------
    InputParameters
        Samples with a leading ``n_samples`` axis; sample 0 equals
        ``params``.
    """
    noise = _sample(
        init_fn=initializers.normal(stddev=1.0),
        n_samples=n_samples,
        key=key,
        params=params,
    )

    def around(x: jax.Array, z: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        samples = x + std * jnp.maximum(jnp.abs(x), 1.0) * z
        return samples.at[0].set(x)

    return jax.tree.map(around, params, noise)


def xavier_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
) -> InputParameters:
    """
    Perform Xavier normal sampling for a given PyTree of parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility.
    params : PyTree
        Parameters to sample (used for shape reference).
    n_samples : int
        Number of samples to generate.
    Returns
    -------
    InputParameters
        Randomly sampled parameters using Xavier normal initialization.
    """
    return _sample(
        init_fn=initializers.xavier_normal(),
        n_samples=n_samples,
        key=key,
        params=params,
    )


def constant_sampling(
    key: PRNGKeyArray,
    params: PyTree,
    n_samples: int,
    value: float = 0.0,
) -> InputParameters:
    """
    Perform constant sampling for a given PyTree of parameters.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key, accepted for interface parity with the other samplers
        but unused -- constant sampling is deterministic.
    params : PyTree
        Parameters to sample (used for shape reference).
    n_samples : int
        Number of samples to generate.
    value : float, optional
        Constant value to fill the samples with, by default 0.0.

    Returns
    -------
    InputParameters
        Parameters filled with the constant value.
    """
    return _sample(
        init_fn=initializers.constant(value),
        n_samples=n_samples,
        key=key,
        params=params,
    )


def grid_sampling(
    params: PyTree,
    n_points_per_dim: int,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> InputParameters:
    """Generate grid samples over parameter space.

    Parameters
    ----------
    params : PyTree
        Parameters to sample (used for shape reference).
    n_points_per_dim : int
        Number of points per dimension in the grid.
    lower_bound : float, optional
        Lower bound for the grid, by default 0.0.
    upper_bound : float, optional
        Upper bound for the grid, by default 1.0.

    Returns
    -------
    InputParameters
        Grid-sampled parameters covering the parameter space.
    """
    grid_params = []
    for param in jax.tree_util.tree_leaves(params):
        # Scalar parameter
        if param.ndim == 0:
            grid = jnp.linspace(lower_bound, upper_bound, num=n_points_per_dim)
        # Tensor parameter
        else:
            grid = jnp.meshgrid(
                *[jnp.linspace(lower_bound, upper_bound, num=n_points_per_dim)]
                * param.size
            )
            # Reshape into parameter shape
            grid = jnp.stack(grid, axis=-1).reshape((-1,) + param.shape)
        grid_params.append(grid)
    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(params), grid_params
    )


# =============================================================================

SAMPLER_MAPPING: dict[str, Callable[..., InputParameters]] = {
    "random": random_sampling,
    "normal": normal_sampling,
    "relative_normal": relative_normal_sampling,
    "xavier": xavier_sampling,
    "constant": constant_sampling,
    "grid": grid_sampling,
}


def get_sampler(name: str) -> Callable[..., InputParameters]:
    """Get a sampler function by name.

    Parameters
    ----------
    name : str
        Name of the sampler to retrieve. Must be one of "random",
        "normal", "relative_normal", "xavier", "constant", or "grid".

    Returns
    -------
    Callable[..., InputParameters]
        The corresponding sampler function.

    Raises
    ------
    ValueError
        If the provided name does not correspond to a valid sampler.
    """
    if name not in SAMPLER_MAPPING:
        raise ValueError(
            f"Sampler '{name}' is not recognized. "
            f"Valid options are: {list(SAMPLER_MAPPING.keys())}"
        )
    return SAMPLER_MAPPING[name]
