import numpy as np
from mace.calculators import mace_mp
from mace.data.utils import Configuration

from phonotune.alexandria.pair_constructor import ConfigSequence, PairConstructor
from phonotune.alexandria.phonon_data import PhononData
from phonotune.structure_utils import convert_configuration_to_ase

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


def main():
    device = "cuda"
    DATA_DIR = "data"
    mp_id = "mp-531340"
    data = PhononData.load_phonon_data(
        mp_id
    )  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
    pc = PairConstructor(
        data
    )  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
    pairs = pc.construct_all_pairs()
    print(f"{len(pairs)} pairs")

    config_seq = ConfigSequence(pairs)

    (train_seq, valid_seq) = config_seq.train_validation_split(0.8)
    valid_seq.to_HDF5(h5_file=f"{DATA_DIR}/mace_multiconfig_mgal2o4_validation.hdf5")
    unrolled_valid_split = valid_seq.unroll()

    calc = mace_mp("small", device)

    rmses = []
    for configuration in unrolled_valid_split:
        ase_atoms = convert_configuration_to_ase(configuration, calc)
        ase_atoms.calc = calc
        mace_forces = ase_atoms.get_forces()

        dft_forces = ase_atoms.get_array("DFT_forces")

        rmse_forces = np.sqrt(np.mean((mace_forces - dft_forces) ** 2)) * 1000
        rmses.append(rmse_forces)

    print(np.mean(np.array(rmses)))


if __name__ == "__main__":
    main()
