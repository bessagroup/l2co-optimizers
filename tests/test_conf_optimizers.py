"""End-to-end tests for the optimizer configs shipped at
``l2co_optimizers/conf/optimizers/``.

Each YAML is a list of ``OptimizationStep`` specs. The tests load every
shipped YAML and check that the *wiring* is right: the ``_target_``
instantiates, every named optimizer is in the registry, the group
composes through ``pkg://l2co_optimizers.conf`` the way consumers use it,
and :func:`l2co_optimizers.create_schedules_experimentdata` builds a row
per step.

The schedule names and hashes are locked against a golden file recorded
from l2co at ``f677d8c``, before the package split: they are the
databank keys, and a config move that silently renamed a schedule would
orphan every stored trajectory under it.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf

from l2co_optimizers import (
    OptimizationStep,
    create_schedules_experimentdata,
    optimizer_mapping,
)

_CONF = resources.files("l2co_optimizers.conf") / "optimizers"
_CONFIGS = sorted(
    p.name.removesuffix(".yaml")
    for p in _CONF.iterdir()
    if p.name.endswith(".yaml")
)
_GOLDEN = json.loads(
    (
        Path(__file__).parent / "data" / "schedule_names_l2co_f677d8c.json"
    ).read_text()
)


def _load(name: str) -> list[OptimizationStep]:
    raw = OmegaConf.load(str(_CONF / f"{name}.yaml"))
    return instantiate(raw, _convert_="all")


def test_ships_the_expected_configs():
    assert set(_CONFIGS) == set(_GOLDEN)


@pytest.mark.parametrize("name", _CONFIGS)
def test_config_instantiates_to_registered_steps(name):
    steps = _load(name)
    assert steps and all(isinstance(s, OptimizationStep) for s in steps)
    for step in steps:
        optimizer_mapping(step.optimizer)  # raises if unregistered


@pytest.mark.parametrize("name", _CONFIGS)
def test_schedule_names_and_hashes_match_pre_split_l2co(name):
    got = [[s.name, s.hash] for s in _load(name)]
    assert got == _GOLDEN[name]


@pytest.mark.parametrize("name", ["medium", "headroom4"])
def test_composes_through_the_package_searchpath(name):
    with initialize_config_module(
        config_module="l2co_optimizers.conf", version_base=None
    ):
        cfg = compose(overrides=[f"+optimizers={name}"])
    steps = instantiate(cfg.optimizers, _convert_="all")
    assert [s.name for s in steps] == [n for n, _ in _GOLDEN[name]]


@pytest.mark.requires_f3dasm
def test_create_schedules_experimentdata_has_a_row_per_step(tmp_path):
    raw = OmegaConf.load(str(_CONF / "medium.yaml"))
    data = create_schedules_experimentdata(raw, tmp_path)
    assert len(data) == len(_GOLDEN["medium"])
