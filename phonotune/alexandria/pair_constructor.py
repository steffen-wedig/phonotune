from dataclasses import dataclass

import numpy as np

from phonotune.alexandria.phonon_data import Displacement, PhononData


@dataclass
class Structure:
    atoms: list[str]
    positions: np.ndarray
    forces: np.ndarray


class PairConstructor:
    def __init__(self, phonon_data: PhononData):
        self.displacements = phonon_data.displacements
        self.supercell = phonon_data.supercell

    def construct_all_pairs(self):
        atom_types = self.supercell.atom_types
        equilibrium_structure = self.supercell.get_positions()
        structure_pairs: list[tuple[Structure, Structure] | None] = []
        number_of_displacements = len(self.displacements)
        i = 0
        while i < number_of_displacements:
            if i + 1 < number_of_displacements and self.displacements[i].is_mirrored(
                self.displacements[i + 1]
            ):
                displacement0 = self.displacements[i]
                displacement1 = self.displacements[i + 1]
                i += 2  # Skip the next vector since we've already processed it.
            else:
                # No mirror present; add v and its mirror.
                displacement0 = self.displacements[i]
                displacement1 = self.displacements[i].get_mirror()
                i += 1

            structure_pairs.append(
                PairConstructor.get_pair(
                    atoms=atom_types,
                    eq_structure=equilibrium_structure,
                    displacement0=displacement0,
                    displacement1=displacement1,
                )
            )

        return structure_pairs

    @staticmethod
    def get_pair(
        atoms, eq_structure, displacement0: Displacement, displacement1: Displacement
    ) -> tuple[Structure, Structure]:
        pos0 = eq_structure
        pos1 = eq_structure

        pos0[displacement0.atom, :] += displacement0.displacement
        pos1[displacement1.atom, :] += displacement1.displacement

        # TODO: Check whether the positions should be wrapped around? Displacement can lead to negative coordinates in the supercell

        if displacement1.forces is None:
            normalized_d0 = displacement0.displacement / np.linalg.norm(
                displacement0.displacement
            )
            forces1 = displacement0.forces - 2 * (displacement0.forces @ normalized_d0)
            displacement1.forces = forces1

        structure0 = Structure(atoms=atoms, positions=pos0, forces=displacement0.forces)
        structure1 = Structure(atoms=atoms, positions=pos1, forces=displacement1.forces)

        return (structure0, structure1)
