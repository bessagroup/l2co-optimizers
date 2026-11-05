"""Plain config values for settings that are functions or objects.

Some settings are not numbers on the optimizer. An evosax algorithm's
``std_schedule`` is an optax schedule passed to its constructor; ARS,
ASEBO, NRES and PES take their mean optimizer, ``optax.adam(1e-3)``, as a
constructor argument; Adafactor's paper learning rate is a function of the
step. A Hydra config cannot hold a function, and a function inside an
``OptimizationStep``'s hyperparameters would put a memory address into its
name and hash, which are databank keys.

So configs give plain values, and the factories convert them here:

- ``std_schedule: 0.2`` becomes ``optax.constant_schedule(0.2)``, for the
  evosax algorithms whose constructor takes ``std_schedule``.
- ``learning_rate: 0.01`` on ARS, ASEBO, NRES or PES becomes their inner
  ``optax.adam(0.01)``.
- ``learning_rate: relative_step`` on Adafactor becomes the paper's relative
  step ``min(1e-2, 1/sqrt(t))``.

Names stay plain (``openes_std_schedule=0.2``,
``adafactor_learning_rate=relative_step``). Spec decision O10 of the dataset
paper's protocol revision.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

# Third-party
import jax
import jax.numpy as jnp
import optax

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Registry names whose evosax constructor takes ``optimizer`` with the
#: default ``optax.adam(1e-3)``; ``learning_rate`` sets that Adam's rate.
#: OpenES and Guided ES use ``optax.sgd`` and ESMC a decaying Adam, which
#: the protocol keeps as they are, so ``learning_rate`` is refused there.
INNER_ADAM: frozenset[str] = frozenset(
    {"ars", "asebo", "noisereusees", "persistentes"}
)


def relative_step(count: jax.Array | int) -> jax.Array:
    """Adafactor's relative step size, ``min(1e-2, 1/sqrt(t))``.

    Shazeer & Stern (2018), with ``t = count + 1``, the 1-based step. Used as
    Adafactor's ``learning_rate``; with ``multiply_by_parameter_scale``
    optax then scales it by ``max(eps, RMS(x))``, as in the paper.

    Parameters
    ----------
    count : jax.Array or int
        The 0-based step count optax passes to a schedule.

    Returns
    -------
    jax.Array
        The relative step size.
    """
    t = jnp.asarray(count, dtype=float) + 1.0
    return jnp.minimum(1e-2, 1.0 / jnp.sqrt(t))


#: Named learning rates: the string in a config -> (schedule, the registry
#: names that may use it, its nominal value). The nominal value stands in
#: where a single number is needed, such as the state-transfer scale.
NAMED_LEARNING_RATES: dict[str, tuple[Callable, frozenset[str], float]] = {
    "relative_step": (relative_step, frozenset({"adafactor"}), 1e-2),
}


def _number(value: Any, key: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{name}: {key} must be a number, got {value!r}. It is "
            f"converted to the optimizer's own object (spec O10)."
        )
    return float(value)


def split_evosax_hyperparameters(
    name: str, algo_cls: type, hyperparameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split plain constructor values off an evosax entry's hyperparameters.

    Parameters
    ----------
    name : str
        Normalised registry name (e.g. ``"openes"``).
    algo_cls : type
        The evosax algorithm class.
    hyperparameters : dict[str, Any]
        The entry's hyperparameters. Not modified.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(constructor_kwargs, rest)``: ``std_schedule`` as a constant
        schedule and, for :data:`INNER_ADAM`, ``optimizer`` as an Adam at
        ``learning_rate``; ``rest`` goes to ``default_params.replace``.

    Raises
    ------
    ValueError
        If ``std_schedule`` is given for an algorithm whose constructor has
        none, ``learning_rate`` for one outside :data:`INNER_ADAM`, or
        either is not a number.
    """
    rest = dict(hyperparameters)
    constructor_kwargs: dict[str, Any] = {}
    accepted = inspect.signature(algo_cls.__init__).parameters
    if "std_schedule" in rest:
        if "std_schedule" not in accepted:
            raise ValueError(
                f"{name}: std_schedule is not a constructor argument of "
                f"{algo_cls.__name__}; set its width through its parameters "
                f"(e.g. std_init) instead."
            )
        width = _number(rest.pop("std_schedule"), "std_schedule", name)
        constructor_kwargs["std_schedule"] = optax.constant_schedule(width)
    if "learning_rate" in rest:
        if name not in INNER_ADAM:
            raise ValueError(
                f"{name}: learning_rate sets the inner Adam only for "
                f"{sorted(INNER_ADAM)}; {algo_cls.__name__} keeps its own "
                f"inner optimizer."
            )
        rate = _number(rest.pop("learning_rate"), "learning_rate", name)
        constructor_kwargs["optimizer"] = optax.adam(learning_rate=rate)
    return constructor_kwargs, rest


def resolve_named_learning_rate(
    name: str, hyperparameters: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn a named ``learning_rate`` into its schedule for an optax entry.

    Parameters
    ----------
    name : str
        Normalised registry name (e.g. ``"adafactor"``).
    hyperparameters : dict[str, Any]
        The entry's hyperparameters. Not modified.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(for_constructor, for_transfer)``: the hyperparameters with the
        schedule, for the optax constructor, and with the schedule's nominal
        number, for the state-transfer builders, which need one number.
        Both are ``hyperparameters`` unchanged when ``learning_rate`` is not
        a string.

    Raises
    ------
    ValueError
        If the name is unknown, or not allowed for this optimizer.
    """
    rate = hyperparameters.get("learning_rate")
    if not isinstance(rate, str):
        return hyperparameters, hyperparameters
    if rate not in NAMED_LEARNING_RATES:
        raise ValueError(
            f"{name}: unknown named learning_rate {rate!r}; known: "
            f"{sorted(NAMED_LEARNING_RATES)}."
        )
    schedule, allowed, nominal = NAMED_LEARNING_RATES[rate]
    if name not in allowed:
        raise ValueError(
            f"{name}: learning_rate {rate!r} is defined only for "
            f"{sorted(allowed)}."
        )
    return (
        {**hyperparameters, "learning_rate": schedule},
        {**hyperparameters, "learning_rate": nominal},
    )
