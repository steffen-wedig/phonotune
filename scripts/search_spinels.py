from phonotune.alexandria.materials_iterator import (
    FileMaterialsIterator,
    ListMaterialsIterator,
)
from phonotune.alexandria.pair_constructor import ConfigFactory
from phonotune.alexandria.phonon_data import PhononDataset
from phonotune.structure_utils import get_spinel_group_mpids

mat_iterator = FileMaterialsIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/alexandira_ph_mpids.txt"
)

spinels, formulas = get_spinel_group_mpids(mat_iterator)

spinel_mpids = [i.string for i in spinels]

print(spinel_mpids)
print(formulas)

dataset = PhononDataset.load_phonon_dataset(
    ListMaterialsIterator(spinel_mpids)
)  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
pc = ConfigFactory(
    dataset.phonon_data_samples
)  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
pairs = pc.construct_all_pairs()

print(len(pairs))
