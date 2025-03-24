import matplotlib.pyplot as plt
from mace.calculators import MACECalculator

from phonotune.phonon_calculation.plotting_bands import (
    plot_model_reference_phonon_comparison,
)
from phonotune.phonon_data.equilibrium_structure import Unitcell
from phonotune.phonon_data.phonon_data import PhononData
from phonotune.structure_utils import unitcell_fire_relaxation

spinel_mpid = "mp-3536"
omat_model_file = "/data/fast-pc-06/snw30/projects/models/mace-omat-0-medium.model"
calc = MACECalculator(omat_model_file, device="cuda", enable_cueq=True)

calc = MACECalculator(
    "/data/fast-pc-06/snw30/projects/phonons/training/mace_single_force_finetune_config_2000/spinel_single_force_finetune_2000.model",
    device="cuda",
    enable_cueq=True,
)


reference_phonons = PhononData.create_phonopy_phonon_from_reference_alexandria_data(
    spinel_mpid
)

unitcell = Unitcell.from_alexandria(spinel_mpid)
unitcell = unitcell_fire_relaxation(unitcell, calc, relaxation_tolerance=0.0001)

phonons, _ = PhononData.create_phonopy_phonon_from_unitcell(unitcell, calc)

phonons.auto_band_structure()

reference_phonons.auto_band_structure()


fig, ax = plt.subplots()

plot_model_reference_phonon_comparison(
    ax, phonons.band_structure, ref_bandstructure=reference_phonons.band_structure
)

plt.savefig("phononbands.pdf")
