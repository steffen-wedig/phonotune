from dataclasses import dataclass

import numpy as np
from ase import Atoms

from phonotune.alexandria.data_utils import (
    download_and_unpack_phonons,
    from_yaml,
    to_yaml,
)


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
    atom_types: list[str]

    @classmethod
    def from_lattice_and_points(cls, lattice, points):
        N_atoms = len(points)
        frac_coordinates = np.zeros(shape=(N_atoms, 3))
        atom_types = []

        for idx, p in enumerate(points):
            frac_coordinates[idx, :] = p["coordinates"]
            atom_types.append(p["symbol"])

        return cls(
            lattice=lattice,
            fractional_coordinates=frac_coordinates,
            atom_types=atom_types,
        )

    def get_positions(self):
        # Multiply fractional coordinates by the lattice matrix. (row vectors are lattice vectors)
        # This converts fractional coordinates (relative units) into Cartesian coordinates.
        return self.fractional_coordinates @ self.lattice


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
            lattice=data["supercell"]["lattice"], points=data["supercell"]["points"]
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

    def get_ase_atoms_from_supercell(self):
        a = Atoms(
            symbols=self.supercell.atom_types,
            scaled_positions=self.supercell.fractional_coordinates,
            cell=self.supercell.lattice,
            pbc=True,
        )

        return a
