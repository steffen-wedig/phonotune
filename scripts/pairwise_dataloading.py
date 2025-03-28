from mace.calculators import mace_mp
from mace.data.hdf5_dataset import MultiConfigHDF5Dataset
from mace.modules.loss import force_difference_mse_error
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import DataLoader

from phonotune.configuration import Configuration
from phonotune.phonon_data.configuration_data import (
    ConfigFactory,
    ConfigSequenceDataset,
)
from phonotune.phonon_data.phonon_data import PhononData

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


# This script requires code that is only present in the unmerged mace fork on https://github.com/steffen-wedig/mace/refactor_data


def main():
    device = "cuda"
    DATA_DIR = "data"
    mp_id = "mp-10499"
    mat_formula = "LiZr2(PO4)3"
    data = PhononData.load_phonon_data(
        mp_id
    )  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
    pc = ConfigFactory(
        data
    )  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
    pairs, _ = pc.construct_configs_per_phonon_data(data)
    print(f"{len(pairs)} pairs")

    config_seq = ConfigSequenceDataset(pairs)

    (train_seq, valid_seq) = config_seq.train_validation_split(0.8)
    valid_seq.to_hdf5(
        h5_file=f"{DATA_DIR}/mace_multiconfig_{mat_formula}_validation.hdf5"
    )

    calc = mace_mp("small", device, default_dtype="float64")

    valid_seq.unroll().to_xyz(
        f"{DATA_DIR}/mace_multiconfig_{mat_formula}_validation.xyz",
        calc,
    )

    N_sample_splits = [20, 50, 100]

    splits = train_seq.get_splits(N_sample_splits)

    for N, split in zip(N_sample_splits, splits, strict=False):
        split.to_hdf5(
            h5_file=f"{DATA_DIR}/mace_multiconfig_{mat_formula}_train_{N}.hdf5"
        )

        unrolled_config_split = split.unroll()

        unrolled_config_split.to_xyz(
            f"{DATA_DIR}/mace_multiconfig_{mat_formula}_train_{N}.xyz",
            calc,
        )

    # This needs to be the atomic number table from MACE Mp
    z_table = get_atomic_number_table_from_zs(
        range(0, 89)
        # symbols2numbers(data.supercell.atom_symbols)
    )  # TODO: Make this nicer

    cutoff = 5.0

    pair_dataset = MultiConfigHDF5Dataset(
        file_path=f"{DATA_DIR}/mace_multiconfig_{mat_formula}_train_20.hdf5",
        r_max=cutoff,
        z_table=z_table,
        config_seq_length=2,
    )

    model = mace_mp("small", device, return_raw_model=True, default_dtpye="float64").to(
        device
    )

    dl = DataLoader(
        pair_dataset, batch_size=1
    )  # This dataloader should return a pair of batches.

    for batch in dl:
        batch.to(device)
        batch_dict = batch.to_dict()
        prediction = model(
            batch_dict,
            training=False,
            compute_force=True,
        )

        loss = force_difference_mse_error(batch, prediction)
        print(loss)


if __name__ == "__main__":
    main()
