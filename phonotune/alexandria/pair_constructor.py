from dataclasses import dataclass

import numpy as np
from ase.symbols import symbols2numbers
from mace.data.utils import Configuration

from phonotune.alexandria.phonon_data import Displacement, PhononData

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


@dataclass
class Structure:
    atom_symbols: list[str]
    positions: np.ndarray
    forces: np.ndarray


class PairConstructor:
    def __init__(self, phonon_data: PhononData):
        self.displacements = phonon_data.displacements
        self.supercell = phonon_data.supercell

    def construct_all_pairs(self) -> ConfigurationPairs:
        atom_symbols = self.supercell.atom_symbols
        equilibrium_structure = self.supercell.get_positions()
        configuration_pairs: ConfigurationPairs = []
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

            configuration_pairs.append(
                PairConstructor.get_pair(
                    atom_symbols=atom_symbols,
                    eq_structure=equilibrium_structure,
                    displacement0=displacement0,
                    displacement1=displacement1,
                )
            )

        return configuration_pairs

    @staticmethod
    def get_pair(
        atom_symbols,
        eq_structure,
        displacement0: Displacement,
        displacement1: Displacement,
    ) -> tuple[Configuration, Configuration]:
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

        atom_numbers = symbols2numbers(atom_symbols)

        # TODO: Maybe calculate force loss weights here?

        config0 = PairConstructor.get_mace_configuration(
            atom_numbers=atom_numbers, positions=pos0, forces=displacement0.forces
        )

        config1 = PairConstructor.get_mace_configuration(
            atom_numbers=atom_numbers, positions=pos1, forces=displacement1.forces
        )

        return (config0, config1)

    @staticmethod
    def get_mace_configuration(atom_numbers, positions, forces) -> Configuration:
        properties = {"forces": forces}
        configuration = Configuration(
            atomic_numbers=atom_numbers,
            positions=positions,
            properties=properties,
            property_weights={"forces": 1.0},
            # property_weights #TODO: define loss weight here! But on the otherhand, the loss weight depends on two configurations.
        )
        return configuration
