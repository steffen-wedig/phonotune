from phonotune.materials_iterator import (
    FileMaterialsIterator,
)
from phonotune.phonon_data.configuration_data import (
    ConfigFactory,
)
from phonotune.phonon_data.phonon_data import PhononDataset


def test_single_construction():
    DATA_DIR = "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels"

    # TRAINING DATA
    mat_iterator = FileMaterialsIterator(f"{DATA_DIR}/train_spinel_mpids")

    pd_dataset = PhononDataset.load_phonon_dataset(mat_iterator)
    pc = ConfigFactory(pd_dataset.phonon_data_samples)

    configs, displacement_data = pc.construct_configs_per_phonon_data(
        pd_dataset.phonon_data_samples[0]
    )
