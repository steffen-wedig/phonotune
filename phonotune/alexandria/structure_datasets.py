from collections.abc import Iterable
from dataclasses import dataclass

import h5py
import numpy as np
from ase.data import chemical_symbols
from ase.symbols import symbols2numbers
from mace.calculators import MACECalculator
from tqdm import tqdm

from phonotune.alexandria.crystal_structures import Supercell, Unitcell
from phonotune.alexandria.data_utils import (
    contains_non_mace_elements,
    is_unstable_lattice,
    open_data,
)
from phonotune.alexandria.materials_iterator import ListMaterialsIterator
from phonotune.structure_utils import unitcell_fire_relaxation


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
            data = open_data(mp_id)

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


@dataclass
class UnitcellDataset:
    """
    This class should load some structures from the alexandria dataset and make them available for analysis.
    """

    unitcells: Iterable[Unitcell]

    @classmethod
    def from_alexandria(
        cls,
        mp_id_iterator: Iterable,
        N_materials: int,
        mace_calculator: MACECalculator,
        skip_unstable: bool = False,
    ):
        unitcells = []
        count = 0
        with tqdm(total=N_materials) as pbar:
            while count < N_materials:
                mp_id = next(mp_id_iterator)
                data = open_data(mp_id)

                if "phonon_freq" not in data:
                    # Some datafiles do not contain phonon frequencies ????
                    continue

                if (
                    is_unstable_lattice(data) or contains_non_mace_elements(data)
                ) and skip_unstable:
                    tqdm.write("Skipped Unstable")
                    continue

                unitcell = Unitcell.from_lattice_and_points(
                    mp_id=mp_id,
                    lattice=data["unit_cell"]["lattice"],
                    points=data["unit_cell"]["points"],
                    phonon_calc_supercell=data["supercell_matrix"],
                    primitive_matrix=data.get(
                        "primitive_matrix", np.eye(3)
                    ),  # This adds the supercell that is needed to calculate phonons later, to ensure that the reference calculcations and our calculations are performed for the same supercell
                )

                try:
                    unitcell = unitcell_fire_relaxation(unitcell, mace_calculator)
                    count = count + 1
                    unitcells.append(unitcell)
                    pbar.update(1)
                except ValueError:
                    tqdm.write("Relaxation did not converge")
                    continue

        return cls(unitcells=unitcells)

    def to_hdf5(self, hdf5_file_path):
        "Converts the supercells to hdf5 storage"

        with h5py.File(hdf5_file_path, "w") as f:
            grp = f.create_group("unitcells")
            for unitcell in self.unitcells:
                subgrp = grp.create_group(name=unitcell.mp_id)
                subgrp["frac_positions"] = unitcell.fractional_coordinates
                subgrp["atomic_numbers"] = symbols2numbers(unitcell.atom_symbols)
                subgrp["lattice"] = unitcell.lattice
                subgrp["supercell_lattice"] = unitcell.phonon_calc_supercell
                subgrp["primitive_matrix"] = unitcell.primitive_matrix

    @classmethod
    def from_hdf5(cls, hdf5_file_path: str):
        "Load from a stored hdf5 file"

        unitcells = []
        with h5py.File(hdf5_file_path, "r") as f:
            grp = f["unitcells"]
            for name in grp:
                frac_positions = f[f"unitcells/{name}/frac_positions"][()]
                atomic_numbers = f[f"unitcells/{name}/atomic_numbers"][()]
                lattice = f[f"unitcells/{name}/lattice"][()]
                supercell_lattice = f[f"unitcells/{name}/supercell_lattice"][()]
                primitive_matrix = f[f"unitcells/{name}/primitive_matrix"][()]

                atomic_symbols = [chemical_symbols[num] for num in atomic_numbers]
                unitcells.append(
                    Unitcell(
                        lattice=lattice,
                        fractional_coordinates=frac_positions,
                        atom_symbols=atomic_symbols,
                        mp_id=name,
                        phonon_calc_supercell=supercell_lattice,
                        primitive_matrix=primitive_matrix,
                    )
                )

        return cls(unitcells)

    def filter_by_atom_symbols(self, atom_symbols: str):
        new_unitcells = []
        for unitcell in self.unitcells:
            # check if given atom symbols is in the
            if atom_symbols in unitcell.atom_symbols:
                new_unitcells.append(unitcell)

        # Overwrite the currently stored data
        self.unitcells = new_unitcells

        return new_unitcells

    def get_materials_iterator(self) -> ListMaterialsIterator:
        mp_ids = []

        for cell in self.unitcells:
            mp_ids.append(cell.mp_id)

        iterator = ListMaterialsIterator(mp_ids)
        return iterator
