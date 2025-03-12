from phonotune.alexandria.data_utils import check_last_atom_displaced
from phonotune.materials_iterator import FileMaterialsIterator

mat_iterator = FileMaterialsIterator(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/rand_alexandria_mp_id.txt"
)


check_last_atom_displaced(mat_iterator)

# max_displacements, mp_id = search_highest_number_displacements(mat_iterator)

# print(f"Max displacements {max_displacements} in mp-id {mp_id}")
