from mace.calculators import mace_mp

from phonotune.alexandria.phonon_data import SupercellDataset
from phonotune.alexandria.structure_iterator import FileStructureIterator

count = 0
structure_iterator = FileStructureIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/alexandira_ph_mpids.txt"
)
N_max = 20
mace_calc = mace_mp("medium", device="cuda", enable_cueq=True)
dataset = SupercellDataset.from_alexandria(structure_iterator, N_max)
dataset_path = (
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/alex_structs.hdf5"
)


dataset.relax_all_atoms(mace_calculator=mace_calc)

dataset.to_hdf5(dataset_path)
