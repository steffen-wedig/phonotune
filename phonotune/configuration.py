from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: np.ndarray  # Angstrom
    properties: dict[str, Any]
    property_weights: dict[str, float]
    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None

    weight: float = 1.0  # weight of config in loss
    config_type: str | None = "Default"  # config_type of config
    head: str | None = "Default"  # head used to compute the config
