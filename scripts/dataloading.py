from mace.calculators import mace_mp
from mace.data.hdf5_dataset import MultiConfigHDF5Dataset
from mace.data.utils import Configuration
from mace.modules.loss import force_difference_mse_error
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import DataLoader

from phonotune.alexandria.pair_constructor import ConfigFactory, ConfigSequence
from phonotune.alexandria.phonon_data import PhononData
from phonotune.structure_utils import configurations_to_xyz

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


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
    pairs, _ = pc.construct_all_pairs()
    print(f"{len(pairs)} pairs")

    config_seq = ConfigSequence(pairs)

    (train_seq, valid_seq) = config_seq.train_validation_split(0.8)
    valid_seq.to_HDF5(
        h5_file=f"{DATA_DIR}/mace_multiconfig_{mat_formula}_validation.hdf5"
    )
    unrolled_valid_split = valid_seq.unroll()

    calc = mace_mp("small", device, default_dtype="float64")

    configurations_to_xyz(
        f"{DATA_DIR}/mace_multiconfig_{mat_formula}_validation.xyz",
        unrolled_valid_split,
        calc,
    )

    N_sample_splits = [20, 50, 100]

    splits = train_seq.get_splits(N_sample_splits)

    for N, split in zip(N_sample_splits, splits, strict=False):
        split.to_HDF5(
            h5_file=f"{DATA_DIR}/mace_multiconfig_{mat_formula}_train_{N}.hdf5"
        )

        unrolled_config_split = split.unroll()

        configurations_to_xyz(
            f"{DATA_DIR}/mace_multiconfig_{mat_formula}_train_{N}.xyz",
            unrolled_config_split,
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
