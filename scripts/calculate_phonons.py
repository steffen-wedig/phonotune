import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from mace.calculators import MACECalculator

from phonotune.helper_functions import aseatoms2phonopy
from phonotune.structure_utils import (
    get_from_mp,
    local_relaxation,
    to_ase,
)
from phonotune.utils_phonon_calculations import calculate_fc2_phonopy_set

MACE_PATH_OMAT = (
    "/home/steffen/projects/phonons/mace_calculators/mace-omat-0-medium.model"
)

MACE_PATH_MP = "/home/steffen/projects/phonons/mace_calculators/2023-12-03-mace-128-L1_epoch-199.model"

mace_calculator = MACECalculator(model_path=MACE_PATH_MP, device="cuda")

struct = get_from_mp("mp-568121")

atoms = to_ase(struct)
local_relaxation(atoms, mace_calculator)
fig, ax = plt.subplots()
plot_atoms(atoms, ax, radii=0.5, rotation=("90x,0y,0z"))
plt.show()

supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
phonon = aseatoms2phonopy(atoms, supercell_matrix)
phonon.generate_displacements(distance=0.03)
supercells = phonon.supercells_with_displacements

force_set = calculate_fc2_phonopy_set(phonon, mace_calculator)
phonon.save()
