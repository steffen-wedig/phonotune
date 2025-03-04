from phonotune.alexandria.data_utils import open_data
from phonotune.alexandria.structure_datasets import UnitcellDataset

unitcell_dataset = UnitcellDataset.from_hdf5(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/unitcell_datasets/bench_unitcell_data.h5"
)
mat_iterator_ref = unitcell_dataset.get_materials_iterator()


for _ in range(0, 5):
    data = open_data(next(mat_iterator_ref))

    print(data["supercell_matrix"])
    print(len(data["primitive_cell"]["points"]))
    print(len(data["unit_cell"]["points"]))


# MACE_MODELS_ROOT = "/data/fast-pc-04/snw30/projects/mace_models"
# MACE_PATH_OMAT = f"{MACE_MODELS_ROOT}/mace-omat-0-medium.model"
#
# N_materials = 2
# unitcell_dataset = UnitcellDataset.from_hdf5("/data/fast-pc-06/snw30/projects/phonons/phonotune/data/unitcell_datasets/bench_unitcell_data.h5")
#
# phonon_dataset_pred = PhononDataset.compute_phonon_dataset_from_unit_cell_dataset(unitcell_dataset=unitcell_dataset, mace_calculator=calc)
#
#
# mat_iterator_ref = unitcell_dataset.get_materials_iterator()
#
# phonon_dataset_ref = PhononDataset.load_phonon_dataset(materials_iterator=mat_iterator_ref, N_materials= N_materials)
