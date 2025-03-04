from phonotune.alexandria.data_utils import search_highest_number_displacements
from phonotune.alexandria.materials_iterator import FileMaterialsIterator

mat_iterator = FileMaterialsIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/alexandira_ph_mpids.txt"
)

max_displacements, mp_id = search_highest_number_displacements(mat_iterator)

print(f"Max displacements {max_displacements} in mp-id {mp_id}")
