from collections.abc import Iterable
from dataclasses import dataclass

import h5py
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.symbols import symbols2numbers
from mace.calculators import MACECalculator
from tqdm import tqdm

from phonotune.alexandria.data_utils import (
    download_and_unpack_phonons,
    from_yaml,
    to_yaml,
)
from phonotune.structure_utils import local_fire_relaxation


@dataclass
class Displacement:
    atom: int
    displacement: np.ndarray
    forces: np.ndarray | None

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
            forces=None,
        )


@dataclass
class Properties:
    energy: np.float64
    entropy: np.float64
    free_energy: np.float64
    heat_capacity: np.float64
    volume: np.float64


@dataclass
class Supercell:
    lattice: np.ndarray
    fractional_coordinates: np.ndarray
    atom_symbols: list[str]
    mp_id: str

    @classmethod
    def from_lattice_and_points(cls, mp_id, lattice, points):
        N_atoms = len(points)
        frac_coordinates = np.zeros(shape=(N_atoms, 3))
        atom_symbols = []

        for idx, p in enumerate(points):
            frac_coordinates[idx, :] = p["coordinates"]
            atom_symbols.append(p["symbol"])

        return cls(
            lattice=lattice,
            fractional_coordinates=frac_coordinates,
            atom_symbols=atom_symbols,
            mp_id=mp_id,
        )

    def get_positions(self):
        # Multiply fractional coordinates by the lattice matrix. (row vectors are lattice vectors)
        # This converts fractional coordinates (relative units) into Cartesian coordinates.
        return self.fractional_coordinates @ self.lattice

    def to_ase_atoms(self) -> Atoms:
        atoms = Atoms(
            symbols=self.atom_symbols,
            scaled_positions=self.fractional_coordinates,
            cell=self.lattice,
            pbc=True,
        )

        return atoms


@dataclass
class PhononData:
    displacements: list[Displacement]
    properties: Properties
    supercell: Supercell
    mp_id: str

    @classmethod
    def load_phonon_data(cls, mp_id):
        # Loads the phonon data either from file or by downloading from the Alexandria website
        try:
            data = from_yaml(mp_id)
        except FileNotFoundError:
            data = download_and_unpack_phonons(mp_id)
            to_yaml(data, mp_id)

        supercell = Supercell.from_lattice_and_points(
            mp_id=mp_id,
            lattice=data["supercell"]["lattice"],
            points=data["supercell"]["points"],
        )
        displacements = [
            Displacement(
                atom=disp["atom"],
                displacement=np.array(disp["displacement"], dtype=np.float64),
                forces=np.array(disp["forces"], dtype=np.float64),
            )
            for disp in data["displacements"]
        ]
        properties = Properties(
            energy=data["energy"],
            entropy=data["entropy"],
            free_energy=data["free_e"],
            heat_capacity=data["heat_capacity"],
            volume=data["volume"],
        )

        return cls(displacements, properties, supercell, mp_id)

    @classmethod
    def calculate_phonon_data_from_supercell(
        cls, supercell: Supercell, mace_calculator: MACECalculator, phonon_data_config
    ):
        pass


@dataclass
class SupercellDataset:
    """
    This class should load some structures from the alexandria dataset and make them available for analysis.
    """

    supercells: Iterable[Supercell]

    @classmethod
    def from_alexandria(cls, mp_id_iterator: Iterable, N_structures: int):
        supercells = []
        count = 0
        while count < N_structures:
            mp_id = next(mp_id_iterator)
            try:
                data = from_yaml(mp_id)
            except FileNotFoundError:
                data = download_and_unpack_phonons(mp_id)
                to_yaml(data, mp_id)

            supercell = Supercell.from_lattice_and_points(
                mp_id=mp_id,
                lattice=data["supercell"]["lattice"],
                points=data["supercell"]["points"],
            )

            supercells.append(supercell)
            count = count + 1

        return cls(supercells=supercells)

    def to_hdf5(self, hdf5_file_path):
        "Converts the supercells to hdf5 storage"

        with h5py.File(hdf5_file_path, "w") as f:
            grp = f.create_group("supercells")
            for supercell in self.supercells:
                subgrp = grp.create_group(name=supercell.mp_id)
                subgrp["frac_positions"] = supercell.fractional_coordinates
                subgrp["atomic_numbers"] = symbols2numbers(supercell.atom_symbols)
                subgrp["lattice"] = supercell.lattice

    @classmethod
    def from_hdf5(cls, hdf5_file_path: str):
        "Load from a stored hdf5 file"

        supercells = []
        with h5py.File(hdf5_file_path, "r") as f:
            grp = f["supercells"]
            for name in grp:
                frac_positions = f[f"supercells/{name}/frac_positions"][()]
                atomic_numbers = f[f"supercells/{name}/atomic_numbers"][()]
                lattice = f[f"supercells/{name}/lattice"][()]

                atomic_symbols = [chemical_symbols[num] for num in atomic_numbers]
                supercells.append(
                    Supercell(
                        lattice=lattice,
                        fractional_coordinates=frac_positions,
                        atom_symbols=atomic_symbols,
                        mp_id=name,
                    )
                )

        return cls(supercells)

    def filter_by_atom_symbols(self, atom_symbols: str):
        new_supercells = []
        for supercell in self.supercells:
            # check if given atom symbols is in the
            if atom_symbols in supercell.atom_symbols:
                new_supercells.append(supercell)

        # Overwrite the currently stored data
        self.supercells = new_supercells

        return new_supercells

    def relax_all_atoms(self, mace_calculator: MACECalculator):
        for supercell in tqdm(self.supercells):
            atoms = supercell.to_ase_atoms()

            local_fire_relaxation(atoms, mace_calculator)
            # Overwrite the field in the supercell

            supercell.fractional_coordinates = atoms.get_scaled_positions()
            supercell.lattice = atoms.get_cell()
