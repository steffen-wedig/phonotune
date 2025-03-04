from mace.calculators import MACECalculator

from phonotune.alexandria.materials_iterator import ListMaterialsIterator
from phonotune.alexandria.model_comparison import ModelComparison
from phonotune.alexandria.phonon_data import PhononDataset
from phonotune.alexandria.structure_datasets import UnitcellDataset

MACE_MODELS_ROOT = "/data/fast-pc-04/snw30/projects/mace_models"

MODEL_NAME = "mace-omat-0-medium"
# MODEL_NAME = "2023-12-03-mace-128-L1_epoch-199"
MACE_PATH = f"{MACE_MODELS_ROOT}/{MODEL_NAME}.model"


calc = MACECalculator(model_path=MACE_PATH, enable_cueq=True, head="default")
reload_dataset = True

mat_iterator = ListMaterialsIterator(["mp-531340"])


unitcell_dataset = UnitcellDataset.from_alexandria(
    mat_iterator, N_materials=1, mace_calculator=calc, skip_unstable=True
)

phonon_dataset_pred = PhononDataset.compute_phonon_dataset_from_unit_cell_dataset(
    unitcell_dataset=unitcell_dataset, mace_calculator=calc
)

mat_iterator = ListMaterialsIterator(["mp-531340"])
phonon_dataset_ref = PhononDataset.load_phonon_dataset(
    materials_iterator=mat_iterator, N_materials=1
)

comp = ModelComparison(dataset_ref=phonon_dataset_ref, dataset_pred=phonon_dataset_pred)

mae_dict = comp.calculate_MAE()
mse, freq, errors = comp.compare_datasets_phonons()
print(mae_dict)
filepath = f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/MgAl2O4_phonon_comp_{MODEL_NAME}.yaml"
comp.store_data(filepath, calculator_name=MODEL_NAME)
