"""
Module for batching data for optimization processes
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Int, PRNGKeyArray, PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class BatchState(eqx.Module):
    """State object for iterating over a dataset in fixed-size batches.

    This class maintains the permutation of dataset indices, the current
    position in that permutation, and provides logic to return the next
    batch of indices while seamlessly handling wrap-around by generating
    a new permutation when the end of the dataset is reached.

    Attributes
    ----------
    perm : Int[Array, " N"]
        1D array containing a permutation of the dataset indices
        (shape: ``(dataset_size,)``). Determines the iteration order.
    pos : int
        Current position (start index) inside ``perm`` for the next batch.
    batch_size : int
        Number of samples per batch. Static field (not traced).
    dataset_size : int
        Total number of samples in the dataset. Static field (not traced).

    Class Methods
    -------------
    init(key, dataset_size, batch_size)
        Construct an initial ``BatchState`` with a fresh permutation and
        position set to zero.

    Properties
    ----------
    dataset_size : int
        Total number of samples in the dataset (``perm.shape[0]``).

    Methods
    -------
    next_batch(dataset)
        Return a batch extracted from the PyTree ``dataset`` and the updated
        ``BatchState``. Wraps around and re-permutes when at end.

    Notes
    -----
    The batching logic chooses between two paths:
    ``take_no_wrap`` when the remaining portion of ``perm`` is large enough
    for a full batch, otherwise ``take_wrap`` which concatenates the tail of
    the current permutation with the head of a newly generated permutation.

    Examples
    --------
    >>> key = jr.PRNGKey(0)
    >>> state = BatchState.init(key, dataset_size=10, batch_size=4)
    >>> data = jnp.arange(10)
    >>> batch, state = state.next_batch(data)
    >>> batch.shape
    (4,)
    """

    perm: Int[Array, " N"]
    pos: Int[Array, ""]
    batch_size: int = eqx.field(static=True)
    dataset_size: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        dataset: PyTree,
        batch_size: int | None,
        key: PRNGKeyArray,
    ):
        """Initialize a ``BatchState`` with a fresh permutation.

        Parameters
        ----------
        dataset : PyTree
            PyTree dataset from which to determine the dataset size.
        batch_size : int | None
            Number of items per batch.
        key : PRNGKeyArray
            PRNG key used to generate the initial permutation and to seed
            subsequent wrap-around permutations.

        Notes
        -----
        If ``dataset_size`` is ``None``, it defaults to zero. If
        ``batch_size`` is ``None``, it defaults to ``dataset_size``.

        Returns
        -------
        BatchState
            New batching state with position set to zero and ``perm`` a
            random permutation of ``range(dataset_size)``.

        Examples
        --------
        >>> key = jr.PRNGKey(0)
        >>> state = BatchState.init(key, dataset_size=10, batch_size=4)
        >>> state.pos
        0
        """
        dataset_size = jax.tree.leaves(dataset)[0].shape[0] if dataset else 0
        batch_size = batch_size if batch_size is not None else dataset_size

        return cls(
            perm=jr.permutation(key, dataset_size),
            pos=jnp.array(0, dtype=int),
            batch_size=batch_size,
            dataset_size=dataset_size,
        )

    def reset(self, key: PRNGKeyArray) -> BatchState:
        """Reset the batching state to initial conditions.

        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key used to generate the initial permutation.

        Returns
        -------
        BatchState
            New batching state with position set to zero and ``perm`` a
            random permutation of ``range(dataset_size)``.
        """
        return BatchState(
            perm=jr.permutation(key, self.dataset_size),
            pos=jnp.array(0, dtype=int),
            batch_size=self.batch_size,
            dataset_size=self.dataset_size,
        )

    def next(self, key: PRNGKeyArray) -> tuple[Int[Array, " N"], BatchState]:
        """Return batch indices and the updated state.

        Parameters
        ----------
        key : PRNGKeyArray
            PRNG key used to generate a new permutation if wrap-around
            occurs.

        Returns
        -------
        tuple[Int[Array, " N"], BatchState]
            A tuple ``(batch_idxs, new_state)`` where ``batch_idxs`` is a
            1D array of length ``batch_size`` containing dataset indices,
            and ``new_state`` is the updated ``BatchState``.
        """

        # Compute the predicate
        pred = (self.pos + self.batch_size) <= self.dataset_size

        # Check if pred is scalar or batched
        # Scalar
        if pred.shape == ():
            return _next_single(self, key)
        # Batched
        else:
            return jax.vmap(_next_single, in_axes=(0, None))(self, key)


# Helper function for a single state
def _next_single(s: BatchState, k: PRNGKeyArray):
    new_state, batch_idxs = jax.lax.cond(
        (s.pos + s.batch_size) <= s.dataset_size,
        take_no_wrap,
        take_wrap,
        (s, k),
    )
    return batch_idxs, new_state


def take_no_wrap(carry: tuple[BatchState, PRNGKeyArray]):
    """Return next batch indices without wrap-around.

    Extracts ``batch_size`` consecutive indices from ``state.perm`` starting
    at ``state.pos`` and advances the position by ``batch_size``. No new
    permutation is generated.

    Parameters
    ----------
    carry : tuple[BatchState, PRNGKeyArray]
        A tuple containing the current batching state and PRNG key.

    Returns
    -------
    tuple[BatchState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` has an
        advanced ``pos`` and ``batch_idxs`` is a 1D array of length
        ``batch_size`` containing dataset indices.
    """
    state, _ = carry
    batch = jax.lax.dynamic_slice(
        state.perm,
        (state.pos,),
        (state.batch_size,),
    )
    new_state = BatchState(
        perm=state.perm,
        pos=state.pos + state.batch_size,
        batch_size=state.batch_size,
        dataset_size=state.dataset_size,
    )
    return new_state, batch


def take_wrap(carry: tuple[BatchState, PRNGKeyArray]):
    """Return next batch indices with wrap-around and re-permutation.

    When the remaining portion of ``state.perm`` is insufficient for a full
    batch, this function takes the tail from the current permutation and the
    head from a freshly generated permutation to form a complete batch. It
    also updates the PRNG key and resets the position within the new
    permutation.

    Parameters
    ----------
    carry : tuple[BatchState, PRNGKeyArray]
        A tuple containing the current batching state and PRNG key.

    Returns
    -------
    tuple[BatchState, jax.Array]
        A tuple ``(new_state, batch_idxs)`` where ``new_state`` contains a
        new permutation and the position set to the number of wrapped
        elements; ``batch_idxs`` is a 1D array of length ``batch_size``
        containing dataset indices.

    Notes
    -----
    The returned indices are constructed by concatenating the remaining tail
    of ``state.perm`` with the head of the new permutation (conceptually).
    """
    state, key = carry
    # Indices for current batch
    idx = jnp.arange(state.batch_size)
    mask = idx < (state.dataset_size - state.pos)

    # Slice the remaining elements from current permutation
    raw = jax.lax.dynamic_slice(state.perm, (state.pos,), (state.batch_size,))

    # Generate new permutation for wrap-around
    new_perm = jr.permutation(key, state.dataset_size)
    second = jax.lax.dynamic_slice(new_perm, (0,), (state.batch_size,))

    # Combine slices using mask: first valid part from raw, rest from second
    batch = jnp.where(mask[::-1], raw, second)

    # Compute new position in new permutation
    new_pos = state.batch_size - (state.dataset_size - state.pos)

    new_state = BatchState(
        perm=new_perm,
        pos=new_pos,
        batch_size=state.batch_size,
        dataset_size=state.dataset_size,
    )
    return new_state, batch
