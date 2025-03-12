from mace.data.utils import Configuration

from phonotune.alexandria.configuration_data import (
    ConfigFactory,
    ConfigSingleDataset,
)
from phonotune.alexandria.phonon_data import PhononDataset
from phonotune.materials_iterator import (
    FileMaterialsIterator,
)

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


def main():
    DATA_DIR = "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels"

    # TRAINING DATA
    mat_iterator = FileMaterialsIterator(f"{DATA_DIR}/train_spinel_mpids")

    pd_dataset = PhononDataset.load_phonon_dataset(mat_iterator)
    pc = ConfigFactory(pd_dataset.phonon_data_samples)
    configs = pc.construct_all_single_configs()

    print(f"{len(configs)} configs")

    config_list = ConfigSingleDataset(configs)
    (train_seq, valid_seq) = config_list.train_validation_split(0.8, shuffle=True)

    valid_seq.to_xyz(f"{DATA_DIR}/mace_multiconfig_spinel_validation.xyz")
    N_sample_splits = [50, 100, 500, 1000, 2000]
    splits = train_seq.get_splits(N_sample_splits)

    for N, split in zip(N_sample_splits, splits, strict=False):
        split.to_xyz(f"{DATA_DIR}/mace_multiconfig_spinel_train_{N}.xyz")

    # TEST DATA
    # Random selection from the alexandria phonon database

    mat_iterator = FileMaterialsIterator(
        "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/rand_alexandria_mp_id.txt"
    )
    pd_dataset = PhononDataset.load_phonon_dataset(mat_iterator, N_materials=25)

    test_mpids = pd_dataset.get_mp_ids()
    with open(
        "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/general_test_alexandria_mpids",
        "w",
    ) as f:
        for mpid in test_mpids:
            f.write(mpid + "\n")

    pc = ConfigFactory(pd_dataset.phonon_data_samples)
    configs = pc.construct_all_single_configs()
    config_list = ConfigSingleDataset(configs)

    config_list.to_xyz(f"{DATA_DIR}/general_test_configs.xyz")
    config_list.to_hdf5(f"{DATA_DIR}/general_test_configs.h5")


if __name__ == "__main__":
    main()
