import h5py
from mace.calculators import mace_mp
from mace.data.hdf5_dataset import MultiConfigHDF5Dataset
from mace.data.utils import Configuration
from mace.modules.loss import force_difference_mse_error
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import DataLoader

from phonotune.alexandria.pair_constructor import (
    PairConstructor,
    save_config_sequence_as_HDF5,
)
from phonotune.alexandria.phonon_data import PhononData

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


def main():
    mp_id = "mp-556756"
    data = PhononData.load_phonon_data(
        mp_id
    )  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
    pc = PairConstructor(
        data
    )  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
    pairs = pc.construct_all_pairs()

    # This needs to be the atomic number table from MACE Mp
    z_table = get_atomic_number_table_from_zs(
        range(0, 89)
        # symbols2numbers(data.supercell.atom_symbols)
    )  # TODO: Make this nicer

    cutoff = 5.0

    # Create the HDF5 dataset form the config pairs
    ##open hdf5 file
    with h5py.File("data/mace_multiconfig.hdf5", "w") as h5_file:
        save_config_sequence_as_HDF5(pairs, h5_file=h5_file)
    # Reload the HDF5 datset using the MultiConfigHDF5Dataset

    pair_dataset = MultiConfigHDF5Dataset(
        file_path="data/mace_multiconfig.hdf5",
        r_max=cutoff,
        z_table=z_table,
        config_seq_length=2,
    )

    device = "cuda"
    model = mace_mp("small", device, return_raw_model=True, default_dtype="float64")

    dl = DataLoader(
        pair_dataset, batch_size=10
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
