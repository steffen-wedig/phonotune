from dataclasses import dataclass

from ase import Atoms
from ase.calculators import calculator
from ase.spacegroup.symmetrize import check_symmetry
from mace.calculators import MACECalculator

import phonotune.structure_utils as su

MACE_MODELS_ROOT = "/data/fast-pc-04/snw30/projects/mace_models"
MACE_PATH_OMAT = f"{MACE_MODELS_ROOT}/mace-omat-0-medium.model"
MACE_PATH_MP = f"{MACE_MODELS_ROOT}/2023-12-03-mace-128-L1_epoch-199.model"


@dataclass
class Material:
    name: str
    temperature: str


@dataclass
class CalculationConfig:
    calculator_type: str
    material: Material
    relaxation_tolerance: float | None = None
    rattle: float | None = None

    @classmethod
    def from_yaml(cls, path: str):
        raise NotImplementedError


class CalculationSetup:
    def __init__(self, config: CalculationConfig):
        self.config = config

    def get_calulator(self) -> calculator:
        match self.config.calculator_type:
            case "MACE_OMAT":
                self.calculator = MACECalculator(
                    model_path=MACE_PATH_OMAT, device="cuda"
                )
            case "MACE_MP_0":
                self.calculator = MACECalculator(model_path=MACE_PATH_MP, device="cuda")
            case "CASTEP":
                raise NotImplementedError
            case _:
                raise ValueError("You did not choose a valid calculator type.")

        return self.calculator

    def get_atoms(self) -> Atoms:
        match self.config.material.name:
            case "Mn4Si7":
                if self.config.material.temperature == "low":
                    struct = su.get_from_mp("mp-568121")
                elif self.config.material.temperature == "high":
                    struct = su.get_from_mp("mp-680339")
                else:
                    raise ValueError
            case "Ru2Sn3":
                if self.config.material.temperature == "low":
                    struct = su.get_low_T_Ru2Sn3_structure()
                elif self.config.material.temperature == "high":
                    struct = su.get_from_mp("mp-680677")

        atoms = su.to_ase(struct)

        sym_data_prior = check_symmetry(atoms, symprec=1e-3, verbose=True)

        su.local_relaxation(
            atoms, self.calculator, self.config.rattle, self.config.relaxation_tolerance
        )

        sym_data_post = check_symmetry(atoms, symprec=1e-3, verbose=True)

        assert sym_data_prior.number == sym_data_post.number, AssertionError(
            "Broke symmetry through optimization or rattling"
        )

        return atoms
