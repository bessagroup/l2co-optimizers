"""ExperimentData construction helpers for optimizer schedules."""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
from f3dasm import ExperimentData
from f3dasm.design import Domain
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local
from l2co_optimizers._src.optimizer_schedule import OptimizationStep

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def create_schedules_experimentdata(
    config: DictConfig,
    project_dir: Path,
) -> ExperimentData:
    """Build an :class:`f3dasm.ExperimentData` of optimizer schedules.

    Each row of the returned ExperimentData carries one
    :class:`OptimizationStep` under the ``optimizer`` input column,
    stored on disk via ``OptimizationStep.save`` /
    ``OptimizationStep.load``.

    Parameters
    ----------
    config : DictConfig
        List-shaped Hydra config (an ``omegaconf.ListConfig``) where
        each item is a ``_target_`` spec that instantiates to an
        :class:`OptimizationStep`. The config is passed to
        ``hydra.utils.instantiate(..., _convert_="all")``; the
        resulting list determines the rows of the ExperimentData,
        one ``OptimizationStep`` per row.
    project_dir : Path
        Project directory under which ``OptimizationStep`` files are
        stored. Used to construct the ExperimentData; files are
        written under ``project_dir/experiment_data/optimizer/``.

    Returns
    -------
    ExperimentData
        ExperimentData with one row per optimizer schedule. Not yet
        stored to disk; call ``.store()`` on the result if
        persistence is required.
    """
    domain = Domain()
    domain.add_parameter(
        name="optimizer",
        to_disk=True,
        store_function=OptimizationStep.save,
        load_function=OptimizationStep.load,
    )
    return ExperimentData(
        domain=domain,
        input_data=[
            {"optimizer": opt} for opt in instantiate(config, _convert_="all")
        ],
        project_dir=project_dir,
    )
