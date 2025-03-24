from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from ase.build import make_supercell
from mace.calculators import MACECalculator

from phonotune.materials_iterator import MaterialsIterator
from phonotune.phonon_calculation.phonon_calculations import (
    calculate_forces_phonopy_set,
)
from phonotune.phonon_data.data_utils import (
    get_displacement_dataset_from_alexandria_data_dict,
    open_data,
)
from phonotune.phonon_data.equilibrium_structure import Supercell, Unitcell
from phonotune.phonon_data.structure_datasets import UnitcellDataset

# These classes are wrangling the Alexandria phonon dataset. The ingest the yaml data files, and create from them the objects to subsequently use in the training data creation.


@dataclass
class Displacement:
    atom: int
    displacement: np.ndarray
    forces: np.ndarray | None
    cell: np.ndarray | None

    def is_mirrored(self, other: "Displacement", tol: float = 1e-8) -> bool:
        # First, ensure that we're comparing displacements for the same atom.
        if self.atom != other.atom:
            return False
        # Check if self.displacement is the negative of other.displacement.
        return np.allclose(self.displacement, -other.displacement, atol=tol)

    @classmethod
    def get_mirror(cls, existing_displacement: "Displacement"):
        return cls(
            atom=existing_displacement.atom,
            displacement=-1.0 * existing_displacement.displacement,
            cell=existing_displacement.cell,
            forces=None,
        )


@dataclass
class Properties:
    energy: np.float64
    entropy: np.float64
    free_energy: np.float64
    heat_capacity: np.float64
    volume: np.float64
    temperatures: list
    N_atoms_unitcell: int


@dataclass
class PhononData:
    displacements: list[Displacement]
    properties: Properties
    supercell: Supercell
    mp_id: str
    phonon_spectrum: dict | None
    calculation_source: str

    @classmethod
    def load_phonon_data(cls, mp_id):
        # Loads the phonon data either from file or by downloading from the Alexandria website
        data = open_data(mp_id)

        supercell = Supercell.from_lattice_and_points(
            mp_id=mp_id,
            lattice=data["supercell"]["lattice"],
            points=data["supercell"]["points"],
        )
        displacements = [
            Displacement(
                atom=disp["atom"] - 1,  # Alexandria dataset is presumably 1 indexed
                displacement=np.array(disp["displacement"], dtype=np.float64),
                forces=np.array(disp["forces"], dtype=np.float64),
                cell=data["supercell"]["lattice"],
            )
            for disp in data["displacements"]
        ]
        properties = Properties(
            energy=data["energy"],
            entropy=data["entropy"],
            free_energy=data["free_e"],
            heat_capacity=data["heat_capacity"],
            volume=data["volume"],
            temperatures=[0.0, 75.0, 150.0, 300.0, 600.0],
            N_atoms_unitcell=len(data["unit_cell"]["points"]),
        )

        phonon_spectrum = {"frequencies": data["phonon_freq"]}

        return cls(
            displacements=displacements,
            properties=properties,
            supercell=supercell,
            mp_id=mp_id,
            phonon_spectrum=phonon_spectrum,
            calculation_source="alexandria",
        )

    @classmethod
    def calculate_phonon_data_from_unitcell(
        cls, unitcell: Unitcell, mace_calculator: MACECalculator
    ):
        phonon, mesh_dict = cls.create_phonopy_phonon_from_unitcell(
            unitcell, mace_calculator
        )

        tp_dict = phonon.get_thermal_properties_dict()

        ase_atoms = unitcell.to_ase_atoms()
        unitcell_volume = ase_atoms.get_volume()
        ase_supercell = make_supercell(ase_atoms, P=unitcell.phonon_calc_supercell)
        # Parse the phonopy.dataset into displacements
        displacements = [
            Displacement(
                atom=i["number"],
                displacement=i["displacement"],
                forces=i["forces"],
                cell=ase_supercell.cell,
            )
            for i in phonon.dataset["first_atoms"]
        ]

        # Stack the unitcell

        mp_id = unitcell.mp_id
        supercell = Supercell(
            lattice=ase_supercell.cell,
            fractional_coordinates=ase_supercell.get_scaled_positions(),
            atom_symbols=ase_supercell.get_chemical_symbols(),
            mp_id=mp_id,
        )

        energy = mace_calculator.get_potential_energy(ase_atoms)

        properties = Properties(
            volume=unitcell_volume,
            energy=energy,
            N_atoms_unitcell=len(unitcell.atom_symbols),
            **tp_dict,
        )

        return cls(
            displacements=displacements,
            properties=properties,
            supercell=supercell,
            mp_id=mp_id,
            phonon_spectrum=mesh_dict,
            calculation_source=str(mace_calculator.name),
        )

        # calculate properties

    @staticmethod
    def create_phonopy_phonon_from_unitcell(
        unitcell: Unitcell, mace_calc: MACECalculator
    ):
        phonon = unitcell.to_phonopy()  # Creates the supercells required for phonopy

        # phonon calculations

        phonon.generate_displacements(distance=0.01)

        _ = phonon.supercells_with_displacements
        _ = calculate_forces_phonopy_set(phonon, mace_calc)

        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()
        phonon.run_mesh(
            np.diag(unitcell.phonon_calc_supercell),
            is_gamma_center=True,
            is_mesh_symmetry=False,
        )
        mesh_dict = phonon.get_mesh_dict()

        temperatures = [0.0, 75.0, 150.0, 300.0, 600.0]
        phonon.run_thermal_properties(temperatures=temperatures)

        return phonon, mesh_dict

    @staticmethod
    def create_phonopy_phonon_from_reference_alexandria_data(mp_id):
        data = open_data(mp_id)

        unitcell = Unitcell.from_alexandria(mp_id)
        print(f"Ref Vol{unitcell.to_ase_atoms().get_volume()}")
        phonon = unitcell.to_phonopy()

        phonon.dataset = get_displacement_dataset_from_alexandria_data_dict(data)
        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()

        assert np.allclose(
            phonon.force_constants,
            np.array(data["force_constants"]["elements"]).reshape(
                phonon.force_constants.shape
            ),
        )

        return phonon


class PhononDataset:
    def __init__(self, phonon_data_samples: Iterable[PhononData]):
        self.phonon_data_samples = phonon_data_samples

    @classmethod
    def load_phonon_dataset(
        cls, materials_iterator: MaterialsIterator, N_materials=None
    ):
        count = 0

        phonon_data_samples = []
        while (
            not N_materials or count < N_materials
        ):  # Always evaluates to true if N_materials is not set
            try:
                mp_id = next(materials_iterator)
            except StopIteration:
                break

            new_phonon_data = PhononData.load_phonon_data(mp_id)
            phonon_data_samples.append(new_phonon_data)
            count += 1

        return cls(phonon_data_samples=phonon_data_samples)

    def to_hdf5():
        raise NotImplementedError

    @classmethod
    def compute_phonon_dataset_from_unit_cell_dataset(
        cls, unitcell_dataset: UnitcellDataset, mace_calculator: MACECalculator
    ):
        phonon_data_samples = []
        for unitcell in unitcell_dataset.unitcells:
            phonon_data = PhononData.calculate_phonon_data_from_unitcell(
                unitcell, mace_calculator
            )
            phonon_data_samples.append(phonon_data)

        return cls(phonon_data_samples)

    def get_mp_ids(self):
        mp_ids = []

        for data in self.phonon_data_samples:
            mp_ids.append(data.mp_id)
        return mp_ids

    def get_source(self):
        sources = []

        for data in self.phonon_data_samples:
            sources.append(data.calculation_source)

        if len(set(sources)) != 1:
            raise ValueError(
                "The Phonon Dataset contains values from multiple distinct calculators, which is not the intended behaviour"
            )

        return sources[0]
