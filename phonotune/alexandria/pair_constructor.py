import numpy as np
from ase.symbols import symbols2numbers
from mace.data.atomic_data import AtomicData
from mace.data.utils import Configuration
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.data import DataSequence

from phonotune.alexandria.phonon_data import Displacement, PhononData

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


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
                displacement1 = Displacement.get_mirror(
                    existing_displacement=self.displacements[i]
                )
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


class PairDataset:
    def __init__(self, data: list[tuple[AtomicData, AtomicData]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_configurations(
        cls, configs: ConfigurationPairs, z_table: AtomicNumberTable, cutoff: float
    ):
        data = []

        for config0, config1 in configs:
            atomic_data0 = AtomicData.from_config(
                config0, z_table=z_table, cutoff=cutoff
            )
            atomic_data1 = AtomicData.from_config(
                config1, z_table=z_table, cutoff=cutoff
            )

            data.append(DataSequence(seq=(atomic_data0, atomic_data1)))

        return cls(data=data)


def save_config_sequence_as_HDF5(data: list[tuple[Configuration, ...]], h5_file):
    grp = h5_file.create_group("config_batch_0")
    for sequence_idx, sequence in enumerate(data):
        subgroup_name = f"sequence_{sequence_idx}"
        seq_subgroup = grp.create_group(subgroup_name)

        for config_idx, config in enumerate(sequence):
            config_subgroup_name = f"config_{config_idx}"
            config_subgroup = seq_subgroup.create_group(config_subgroup_name)
            config_subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
            config_subgroup["positions"] = write_value(config.positions)
            properties_subgrp = config_subgroup.create_group("properties")
            for key, value in config.properties.items():
                properties_subgrp[key] = write_value(value)
            config_subgroup["cell"] = write_value(config.cell)
            config_subgroup["pbc"] = write_value(config.pbc)
            config_subgroup["weight"] = write_value(config.weight)
            weights_subgrp = config_subgroup.create_group("property_weights")
            for key, value in config.property_weights.items():
                weights_subgrp[key] = write_value(value)
            config_subgroup["config_type"] = write_value(config.config_type)


def write_value(value):
    return value if value is not None else "None"
