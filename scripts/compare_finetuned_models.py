from mace.calculators import MACECalculator

from phonotune.alexandria.materials_iterator import ListMaterialsIterator
from phonotune.alexandria.model_comparison import (
    ModelComparison,
    calculate_validation_loss,
)
from phonotune.alexandria.pair_constructor import ConfigSequence
from phonotune.alexandria.phonon_data import PhononDataset
from phonotune.alexandria.structure_datasets import UnitcellDataset

MACE_MODELS_ROOT = "/data/fast-pc-06/snw30/projects/models"

material = "mp-10499"
mat_formula = "LiZr2(PO4)3"

model_names = [
    "lizr2po43_single_force_finetune_100",
    "2023-12-03-mace-128-L1_epoch-199",
]


validation_dataset = ConfigSequence.from_HDF5(
    f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/mace_multiconfig_{mat_formula}_validation.hdf5"
)

for model_name in model_names:
    try:
        MACE_PATH = f"{MACE_MODELS_ROOT}/{model_name}.model"

        try:
            calc = MACECalculator(
                model_path=MACE_PATH, enable_cueq=True, head="default"
            )
        except AssertionError:
            calc = MACECalculator(
                model_path=MACE_PATH, enable_cueq=True, head="Default"
            )

        reload_dataset = True

        val_loss = calculate_validation_loss(calc, validation_dataset)
        print(val_loss)

        mat_iterator = ListMaterialsIterator([material])
        unitcell_dataset = UnitcellDataset.from_alexandria(
            mat_iterator, N_materials=1, mace_calculator=calc, skip_unstable=True
        )

        phonon_dataset_pred = (
            PhononDataset.compute_phonon_dataset_from_unit_cell_dataset(
                unitcell_dataset=unitcell_dataset, mace_calculator=calc
            )
        )

        mat_iterator = ListMaterialsIterator([material])
        phonon_dataset_ref = PhononDataset.load_phonon_dataset(
            materials_iterator=mat_iterator, N_materials=1
        )

        comp = ModelComparison(
            dataset_ref=phonon_dataset_ref, dataset_pred=phonon_dataset_pred
        )

        mae_dict = comp.calculate_MAE()
        mse, freq, errors = comp.compare_datasets_phonons()
        print(f"Phonon MSE {mse}")
        filepath = f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/{mat_formula}_phonon_comp_{model_name}.yaml"
        comp.store_data(filepath, calculator_name=model_name, validation_loss=val_loss)
    except Exception as e:
        print(e)
        continue
