import numpy as np
from ase import Atoms
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phonotune.alexandria.data_utils import open_data, unpack_points


class Cell:
    def __init__(self, lattice, fractional_coordinates, atom_symbols, mp_id):
        self.lattice = lattice
        self.fractional_coordinates = fractional_coordinates
        self.atom_symbols = atom_symbols
        self.mp_id = mp_id

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


class Unitcell(Cell):
    def __init__(
        self,
        lattice,
        fractional_coordinates,
        atom_symbols,
        mp_id,
        phonon_calc_supercell,
        primitive_matrix,
    ):
        super().__init__(lattice, fractional_coordinates, atom_symbols, mp_id)
        self.phonon_calc_supercell = phonon_calc_supercell
        self.primitive_matrix = primitive_matrix

    @classmethod
    def from_alexandria(cls, mp_id):
        data = open_data(mp_id)

        unitcell = Unitcell.from_lattice_and_points(
            mp_id=mp_id,
            lattice=data["unit_cell"]["lattice"],
            points=data["unit_cell"]["points"],
            phonon_calc_supercell=data["supercell_matrix"],
            primitive_matrix=data.get("primitive_matrix", np.eye(3)),
        )

        return unitcell

    @classmethod
    def from_lattice_and_points(
        cls, mp_id, lattice, points, phonon_calc_supercell, primitive_matrix
    ):
        frac_coordinates, atom_symbols = unpack_points(points)

        return cls(
            lattice=lattice,
            fractional_coordinates=frac_coordinates,
            atom_symbols=atom_symbols,
            mp_id=mp_id,
            phonon_calc_supercell=phonon_calc_supercell,
            primitive_matrix=primitive_matrix,
        )

    def to_phonopy(self):
        phonoatoms = PhonopyAtoms(
            symbols=self.atom_symbols,
            scaled_positions=self.fractional_coordinates,
            cell=self.lattice,
            pbc=True,
        )
        return Phonopy(
            unitcell=phonoatoms,
            supercell_matrix=self.phonon_calc_supercell,
            primitive_matrix=self.primitive_matrix,
            symprec=1e-5,
        )


class Supercell(Cell):
    def __init__(self, lattice, fractional_coordinates, atom_symbols, mp_id):
        super().__init__(lattice, fractional_coordinates, atom_symbols, mp_id)

    @classmethod
    def from_lattice_and_points(cls, mp_id, lattice, points):
        frac_coordinates, atom_symbols = unpack_points(points)

        return cls(
            lattice=lattice,
            fractional_coordinates=frac_coordinates,
            atom_symbols=atom_symbols,
            mp_id=mp_id,
        )
