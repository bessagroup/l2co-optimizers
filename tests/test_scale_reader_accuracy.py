"""Every gradient optimizer's scale reader matches the step it takes.

A scale reader used to feed only the scale write. Under l2co ADR 0022 it
also decides whether an optimizer's own state still holds at a switch's
destination: the destination's distance is measured in the optimizer's
own widths, so a reader that is 5x off moves that threshold by 5x. The
readers are therefore held to the step the optimizer actually takes:
within 1.5x of it from the very first step, or reporting no scale at
all (l2co-optimizers ADR 0012). An optimizer that reports no scale is
restarted whenever a switch moves it, which is the safe failure.

Before this test, the Adam family's reader skipped bias correction and
overstated the step by 3.2x at step 1 and 6.5x at step 10.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

# Third-party
import jax
import jax.numpy as jnp
import optax
import pytest

# Local
from l2co_optimizers import OptimizationStep, optimizer_parts
from l2co_optimizers._src.core.state_transfer import (
    ADAM_STYLE,
    CONF_ABSENT,
    UNREADABLE_STEP,
)
from l2co_optimizers._src.optax_implementations import (
    normalized_optax_from_state,
    normalized_optax_normal,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

#: Allowed ratio between the reader's scale and the step's RMS.
TOLERANCE = 1.5
#: Steps at which the reader is compared with the step just taken.
CHECKPOINTS = (1, 2, 5, 10, 30)
DIM = 8

#: L-BFGS has its own exact reader and is reset on every switch with
#: news, so its scale never decides a restart.
EXCLUDED = {"lbfgs"}

GRADIENT_OPTIMIZERS = sorted(
    (set(normalized_optax_normal) | set(normalized_optax_from_state))
    - EXCLUDED
)

_H = jnp.logspace(0.0, 2.0, DIM)


def _loss(x):
    return 0.5 * jnp.sum(_H * x**2)


def _parts(name):
    x0 = jnp.linspace(-1.0, 1.0, DIM)
    for hyperparameters in ({"learning_rate": 1e-2}, {}):
        try:
            return optimizer_parts(
                OptimizationStep(
                    optimizer=name, hyperparameters=hyperparameters
                ),
                model=x0,
                loss_fn=lambda model, *args, **kwargs: _loss(model),
                pass_rng=False,
            ), x0
        except TypeError:
            continue
    pytest.skip(f"{name} cannot be built with or without a learning rate")


def _ratios(name):
    """``(step, conf, reader / step RMS)`` at each checkpoint."""
    parts, x = _parts(name)
    transform = parts.transform
    state = transform.init(x)
    out = []
    for step in range(1, max(CHECKPOINTS) + 1):
        grad = jax.grad(_loss)(x)
        try:
            updates, state = transform.update(
                grad, state, x, value=_loss(x), grad=grad, value_fn=_loss
            )
        except TypeError:
            updates, state = transform.update(grad, state, x)
        x = optax.apply_updates(x, updates)
        if step in CHECKPOINTS:
            sigma, conf = parts.transfer_read_fn(state)
            rms = float(jnp.sqrt(jnp.mean(jnp.square(updates))))
            # A step of exactly zero (Rprop's first, before it has a sign
            # to compare against) gives nothing to measure the reader by.
            ratio = float(sigma) / rms if rms > 0.0 else None
            out.append((step, float(conf), ratio))
    return out


@pytest.mark.parametrize("name", GRADIENT_OPTIMIZERS)
def test_reader_matches_the_step_or_reports_none(name):
    for step, conf, ratio in _ratios(name):
        if conf == CONF_ABSENT or ratio is None:
            continue
        assert 1.0 / TOLERANCE <= ratio <= TOLERANCE, (
            f"{name}: at step {step} the reader is {ratio:.3g}x the step "
            f"it actually took"
        )


@pytest.mark.parametrize("name", sorted(ADAM_STYLE))
def test_adam_style_readers_report_a_scale(name):
    # A bias-corrected reader must not pass by reporting nothing.
    assert all(conf > CONF_ABSENT for _, conf, _ in _ratios(name))


@pytest.mark.parametrize("name", sorted(UNREADABLE_STEP))
def test_unreadable_steps_report_none(name):
    assert all(conf == CONF_ABSENT for _, conf, _ in _ratios(name))
