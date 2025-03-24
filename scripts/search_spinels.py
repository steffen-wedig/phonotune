from phonotune.materials_iterator import (
    FileMaterialsIterator,
)
from phonotune.structure_utils import get_spinel_group_mpids

# Script that finds the aluminium oxide dataset from the materials project


mat_iterator = FileMaterialsIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/alexandira_ph_mpids.txt"
)

spinels, formulas, elements = get_spinel_group_mpids(mat_iterator)

spinel_mpids = [i.string for i in spinels]

training_spinels = spinel_mpids[:-3]
test_spinels = spinel_mpids[-3:]

with open(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/train_spinel_mpids",
    "w",
) as f:
    for mpid in training_spinels:
        f.write(mpid + "\n")


with open(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/test_spinel_mpids",
    "w",
) as f:
    for mpid in test_spinels:
        f.write(mpid + "\n")
