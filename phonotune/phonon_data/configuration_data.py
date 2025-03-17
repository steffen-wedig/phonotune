import math
import random
from collections.abc import Sequence

import h5py
import numpy as np
from ase.io.extxyz import write_extxyz
from ase.symbols import symbols2numbers

from phonotune.alexandria.phonon_data import Displacement, PhononData
from phonotune.configuration import Configuration
from phonotune.structure_utils import convert_configuration_to_ase

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


class ConfigFactory:
    def __init__(self, phonon_data: list[PhononData]):
        self.phonon_data_samples = phonon_data

    def construct_all_pairs(self) -> ConfigurationPairs:
        config_pairs_list = []

        for pd in self.phonon_data_samples:
            config_pairs, _ = self.construct_pair_per_phonon_data(pd)
            config_pairs_list.extend(config_pairs)

        return config_pairs_list

    @staticmethod
    def construct_pair_per_phonon_data(phonon_data: PhononData):
        atom_symbols = phonon_data.supercell.atom_symbols
        equilibrium_structure = phonon_data.supercell.get_positions()

        configuration_pairs: ConfigurationPairs = []
        number_of_displacements = len(phonon_data.displacements)
        i = 0

        displacement_data = []

        while i < number_of_displacements:
            if i + 1 < number_of_displacements and phonon_data.displacements[
                i
            ].is_mirrored(phonon_data.displacements[i + 1]):
                displacement0 = phonon_data.displacements[i]
                displacement1 = phonon_data.displacements[i + 1]

                i += 2  # Skip the next vector since we've already processed it.
            else:
                # No mirror present; add v and its mirror.
                displacement0 = phonon_data.displacements[i]
                displacement1 = Displacement.get_mirror(
                    existing_displacement=phonon_data.displacements[i]
                )
                i += 1

            (config1, config2), forces0, forces1 = ConfigFactory.get_pair(
                atom_symbols=atom_symbols,
                eq_structure=equilibrium_structure,
                displacement0=displacement0,
                displacement1=displacement1,
            )
            configuration_pairs.append((config1, config2))

            displacement_data.append(
                {
                    "number": displacement0.atom,
                    "displacement": displacement0.displacement,
                    "forces": forces0,
                }
            )
            displacement_data.append(
                {
                    "number": displacement1.atom,
                    "displacement": displacement1.displacement,
                    "forces": forces1,
                }
            )

        displacement_dataset = {
            "natoms": len(atom_symbols),
            "first_atoms": displacement_data,
        }

        return configuration_pairs, displacement_dataset

    def construct_all_single_configs(self):
        config_list = []

        for pd in self.phonon_data_samples:
            configs, _ = self.construct_configs_per_phonon_data(pd)
            config_list.extend(configs)

        return config_list

    @staticmethod
    def construct_configs_per_phonon_data(phonon_data: PhononData):
        atom_symbols = phonon_data.supercell.atom_symbols
        equilibrium_structure = phonon_data.supercell.get_positions()

        configs = []

        displacement_data = []

        for disp in phonon_data.displacements:
            config = ConfigFactory.convert_displacement_to_config(
                equilibrium_structure, atom_symbols, disp
            )
            configs.append(config)

            displacement_data.append(
                {
                    "number": disp.atom,
                    "displacement": disp.displacement,
                    "forces": disp.forces,
                }
            )

        displacement_dataset = {
            "natoms": len(atom_symbols),
            "first_atoms": displacement_data,
        }

        return configs, displacement_dataset

    @staticmethod
    def convert_displacement_to_config(
        equilibrium_structure, atom_symbols, displacement: Displacement
    ) -> Configuration:
        position = equilibrium_structure.copy()
        position[displacement.atom, :] += displacement.displacement

        config = ConfigFactory.get_mace_configuration(
            atom_numbers=symbols2numbers(atom_symbols),
            positions=position,
            forces=displacement.forces,
            cell=displacement.cell,
        )

        return config

    @staticmethod
    def get_pair(
        atom_symbols,
        eq_structure,
        displacement0: Displacement,
        displacement1: Displacement,
    ) -> tuple[Configuration, Configuration]:
        pos0 = eq_structure.copy()
        pos1 = eq_structure.copy()

        pos0[displacement0.atom, :] += displacement0.displacement
        pos1[displacement1.atom, :] += displacement1.displacement

        # TODO: Check whether the positions should be wrapped around? Displacement can lead to negative coordinates in the supercell

        if displacement1.forces is None:
            normalized_d0 = displacement0.displacement / np.linalg.norm(
                displacement0.displacement
            )

            forces1 = (
                displacement0.forces
                - 2
                * (displacement0.forces @ normalized_d0).reshape(-1, 1)
                * normalized_d0
            )
            displacement1.forces = forces1

        atom_numbers = symbols2numbers(atom_symbols)

        # TODO: Maybe calculate force loss weights here?

        config0 = ConfigFactory.get_mace_configuration(
            atom_numbers=atom_numbers,
            positions=pos0,
            forces=displacement0.forces,
            cell=displacement0.cell,
        )

        config1 = ConfigFactory.get_mace_configuration(
            atom_numbers=atom_numbers,
            positions=pos1,
            forces=displacement1.forces,
            cell=displacement1.cell,
        )

        return (config0, config1), displacement0.forces, displacement1.forces

    @staticmethod
    def get_mace_configuration(atom_numbers, positions, forces, cell) -> Configuration:
        properties = {"DFT_forces": forces, "DFT_energy": 0.0}
        configuration = Configuration(
            atomic_numbers=atom_numbers,
            positions=positions,
            cell=cell,
            properties=properties,
            property_weights={"DFT_forces": 1.0, "DFT_energy": 0.0},
            pbc=(True, True, True),
            # property_weights #TODO: define loss weight here! But on the otherhand, the loss weight depends on two configurations.
        )
        return configuration


class ConfigSingleDataset:
    def __init__(self, data: list[Configuration]):
        self.data = data

    def get_splits(self, N_splits: Sequence[int]) -> tuple["ConfigSingleDataset"]:
        # shuffle???
        splits = []
        for i in N_splits:
            split = self.data[:i]
            splits.append(ConfigSingleDataset(split))
        return splits

    def train_validation_split(
        self, train_valid_ratio: float, shuffle=False
    ) -> tuple["ConfigSingleDataset", "ConfigSingleDataset"]:
        split_index = math.floor(train_valid_ratio * len(self.data))

        if shuffle:
            random.shuffle(self.data)

        training_data = self.data[:split_index]
        validation_data = self.data[split_index:]

        return ConfigSingleDataset(training_data), ConfigSingleDataset(validation_data)

    def to_xyz(self, xyz_file_path: str):
        ase_atoms = []

        for config in self.data:
            atoms = convert_configuration_to_ase(config)
            ase_atoms.append(atoms)

        f = open(xyz_file_path, "w")
        write_extxyz(
            f,
            ase_atoms,
            columns=["symbols", "positions", "DFT_forces"],
            write_info=True,
        )

    def to_hdf5(self, h5_file):
        # This creates hdf5 files based on the hdf5 layout of the mace/refactor_data branch
        with h5py.File(h5_file, "w") as f:
            grp = f.create_group("config_batch_0")

            for config_idx, config in enumerate(self.data):
                config_subgroup_name = f"config_{config_idx}"
                config_subgroup = grp.create_group(config_subgroup_name)
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

    def to_main_hdf5(self, h5_file):
        # This creates hdf5 files based on the hdf5 layout of the mace/main branch
        pass

    @classmethod
    def from_hdf5(cls, h5_file):
        configurations = []

        with h5py.File(h5_file, "r") as f:
            # Iterate over all batch groups (e.g., "config_batch_0", "config_batch_1", etc.)
            for batch_key in sorted(f.keys()):
                if not batch_key.startswith("config_batch_"):
                    continue
                batch_grp = f[batch_key]
                # Iterate over each configuration subgroup in the batch
                for config_key in sorted(batch_grp.keys()):
                    if not config_key.startswith("config_"):
                        continue
                    subgrp = batch_grp[config_key]

                    # Read atomic_numbers and positions directly (assuming they were stored as arrays)
                    atomic_numbers = subgrp["atomic_numbers"][()]
                    positions = subgrp["positions"][()]

                    # Read properties dictionary
                    properties = {}
                    properties_grp = subgrp["properties"]
                    for key in properties_grp:
                        properties[key] = read_value(properties_grp[key][()])

                    # Read remaining attributes using read_value
                    cell = read_value(subgrp["cell"][()])
                    pbc = read_value(subgrp["pbc"][()])
                    weight = read_value(subgrp["weight"][()])

                    # Read property_weights dictionary
                    property_weights = {}
                    weights_grp = subgrp["property_weights"]
                    for key in weights_grp:
                        property_weights[key] = read_value(weights_grp[key][()])

                    config_type = read_value(subgrp["config_type"][()])

                    # Construct the Configuration object.
                    config = Configuration(
                        atomic_numbers=atomic_numbers,
                        positions=positions,
                        properties=properties,
                        property_weights=property_weights,
                        cell=cell,
                        pbc=pbc,
                        weight=weight,
                        config_type=config_type,
                    )
                    configurations.append(config)

        return cls(data=configurations)


class ConfigSequenceDataset:
    def __init__(self, data: list[tuple[Configuration, ...]]):
        self.data = data

    def to_hdf5(self, h5_file):
        with h5py.File(h5_file, "w") as f:
            grp = f.create_group("config_batch_0")
            for sequence_idx, sequence in enumerate(self.data):
                subgroup_name = f"sequence_{sequence_idx}"
                seq_subgroup = grp.create_group(subgroup_name)

                for config_idx, config in enumerate(sequence):
                    config_subgroup_name = f"config_{config_idx}"
                    config_subgroup = seq_subgroup.create_group(config_subgroup_name)
                    config_subgroup["atomic_numbers"] = write_value(
                        config.atomic_numbers
                    )
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

    @classmethod
    def from_hdf5(cls, h5_file):
        data = []
        with h5py.File(h5_file, "r") as f:
            grp = f["config_batch_0"]
            # Sort sequence groups by the numeric index extracted from their names.
            sequence_keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
            for seq_key in sequence_keys:
                seq_subgroup = grp[seq_key]
                sequence_configs = []
                # Similarly, sort configuration groups within each sequence.
                config_keys = sorted(
                    seq_subgroup.keys(), key=lambda x: int(x.split("_")[1])
                )
                for config_key in config_keys:
                    config_subgroup = seq_subgroup[config_key]
                    # Read each stored value using the corresponding read_value function.
                    atomic_numbers = read_value(config_subgroup["atomic_numbers"][()])
                    positions = read_value(config_subgroup["positions"][()])

                    # Read properties stored in a subgroup.
                    properties = {}
                    properties_subgrp = config_subgroup["properties"]
                    for key in properties_subgrp.keys():
                        properties[key] = read_value(properties_subgrp[key][()])

                    cell = read_value(config_subgroup["cell"][()])
                    pbc = read_value(config_subgroup["pbc"][()])
                    weight = read_value(config_subgroup["weight"][()])

                    # Read property weights.
                    property_weights = {}
                    weights_subgrp = config_subgroup["property_weights"]
                    for key in weights_subgrp.keys():
                        property_weights[key] = read_value(weights_subgrp[key])

                    config_type = read_value(config_subgroup["config_type"][()])

                    # Create a Configuration instance.
                    config = Configuration(
                        atomic_numbers=atomic_numbers,
                        positions=positions,
                        properties=properties,
                        cell=cell,
                        pbc=pbc,
                        weight=weight,
                        property_weights=property_weights,
                        config_type=config_type,
                    )
                    sequence_configs.append(config)
                data.append(tuple(sequence_configs))
        return cls(data)

    def train_validation_split(
        self, train_valid_ratio: float
    ) -> tuple["ConfigSequenceDataset", "ConfigSequenceDataset"]:
        split_index = math.floor(train_valid_ratio * len(self.data))

        training_data = self.data[:split_index]
        validation_data = self.data[split_index:]

        return ConfigSequenceDataset(training_data), ConfigSequenceDataset(
            validation_data
        )

    def get_splits(self, N_splits: Sequence[int]) -> tuple["ConfigSequenceDataset"]:
        # shuffle???
        splits = []
        for i in N_splits:
            split = self.data[:i]
            splits.append(ConfigSequenceDataset(split))
        return splits

    def unroll(self):
        unrolled_configs = []
        for configs in self.data:
            unrolled_configs.extend(configs)

        return ConfigSingleDataset(unrolled_configs)


def write_value(value):
    return value if value is not None else "None"


def read_value(value):
    if isinstance(value, str):
        if value == "None":
            return None

    return value
