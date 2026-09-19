"""Intra-node MegaMoE on gfx950, ported from Petit."""

from .api import ExpertWeights, MegaMoeInputViews, MegaMoeWorkspace, dynamic_mxfp4_mega_moe, pack_expert_weights
from .config import MegaMoeConfig
from .dispatch import get_2stage_cfgs, registered_solutions
from .quantization import quantize_input
from .solutionid import ActivationFunction, DataType, MegaMoeSolutionId, ProducerGeometry, W2TileShape

__all__ = [
    "ActivationFunction",
    "DataType",
    "ExpertWeights",
    "MegaMoeConfig",
    "MegaMoeInputViews",
    "MegaMoeSolutionId",
    "MegaMoeWorkspace",
    "ProducerGeometry",
    "W2TileShape",
    "dynamic_mxfp4_mega_moe",
    "get_2stage_cfgs",
    "pack_expert_weights",
    "quantize_input",
    "registered_solutions",
]
