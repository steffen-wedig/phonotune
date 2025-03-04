from mace.calculators import MACECalculator

from phonotune.alexandria.materials_iterator import FileMaterialsIterator
from phonotune.alexandria.model_comparison import ModelComparison
from phonotune.alexandria.phonon_data import PhononDataset
from phonotune.alexandria.structure_datasets import UnitcellDataset

MACE_MODELS_ROOT = "/data/fast-pc-04/snw30/projects/mace_models"

MODEL_NAME = "mace-omat-0-medium"
MACE_PATH = f"{MACE_MODELS_ROOT}/{MODEL_NAME}.model"


N_materials = 20
DATASET_NAME = f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/unitcell_datasets/bench_unitcell_data_{N_materials}_{MODEL_NAME}.h5"
calc = MACECalculator(model_path=MACE_PATH, enable_cueq=True, head="default")
reload_dataset = True


mat_iterator = FileMaterialsIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/rand_alexandria_mp_id.txt"
)


if reload_dataset:
    # create unitcelldataset
    unitcell_dataset = UnitcellDataset.from_alexandria(
        mat_iterator, N_materials=N_materials, mace_calculator=calc, skip_unstable=True
    )
    unitcell_dataset.to_hdf5(DATASET_NAME)
else:
    unitcell_dataset = UnitcellDataset.from_hdf5(DATASET_NAME)

phonon_dataset_pred = PhononDataset.compute_phonon_dataset_from_unit_cell_dataset(
    unitcell_dataset=unitcell_dataset, mace_calculator=calc
)


mat_iterator_ref = unitcell_dataset.get_materials_iterator()

phonon_dataset_ref = PhononDataset.load_phonon_dataset(
    materials_iterator=mat_iterator_ref, N_materials=N_materials
)

comp = ModelComparison(dataset_ref=phonon_dataset_ref, dataset_pred=phonon_dataset_pred)

mae_dict = comp.calculate_MAE()
mse, freq, errors = comp.compare_datasets_phonons()

filepath = f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/phonon_comp_{N_materials}_{MODEL_NAME}.yaml"
comp.store_data(filepath, calculator_name=MODEL_NAME)
