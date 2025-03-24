import matplotlib.pyplot as plt
import numpy as np
from mace.calculators import MACECalculator, mace_mp

from phonotune.evaluation.training_evaluation import (
    get_configurations_across_linear_displacement,
)
from phonotune.phonon_data.equilibrium_structure import Unitcell
from phonotune.phonon_data.phonon_data import PhononData

spinel_mpid = "mp-3536"
spinel_mpid = "mp-531840"

p_data = PhononData.load_phonon_data(spinel_mpid)

displacement = p_data.displacements[50]

ref_force = np.linalg.norm(displacement.forces)

unitcell = Unitcell.from_alexandria(spinel_mpid)
atoms = unitcell.to_ase_atoms()
N_atoms = len(atoms.get_chemical_symbols())
N_displacements = 51

displacement_vec, structures = get_configurations_across_linear_displacement(
    atoms, displacement, N_displacements
)
# displacement_vec, structures = get_configurations_across_orthogonal_displacement(atoms,displacement,N_displacements)

omat_model_file = "/data/fast-pc-06/snw30/projects/models/mace-omat-0-medium.model"
omat_calc = MACECalculator(omat_model_file, device="cuda", enable_cueq=True)


ft_calc_replay = MACECalculator(
    "/data/fast-pc-06/snw30/projects/phonons/training/mace_single_force_finetune_config_w_replay_2000/spinel_single_force_finetune_2000_wrp.model",
    device="cuda",
    enable_cueq=True,
)

ft_calc_no_replay = MACECalculator(
    "/data/fast-pc-06/snw30/projects/phonons/training/mace_single_force_finetune_config_2000/spinel_single_force_finetune_2000.model",
    device="cuda",
    enable_cueq=True,
)

mace_mp_medium_reference_calc = mace_mp(
    "medium", device="cuda", enable_cueq=True, default_dtype="float64"
)


ref_energy = mace_mp_medium_reference_calc.get_potential_energy(atoms)

# hessian_mace_mp = mace_mp_medium_reference_calc.get_hessian(atoms)

# print(hessian_mace_mp.shape)
# harmonic_mace_mp = np.sum(np.diag(np.transpose(hessian_mace_mp, (0, 2, 1)).#reshape((3*N_atoms, 3*N_atoms))))
# print(harmonic_mace_mp)

# hessian_ft = ft_calc_no_replay.get_hessian(atoms)
# harmonic_ft = np.sum(np.diag(np.transpose(hessian_ft, (0, 2, 1)).reshape((3*N_atoms, 3* N_atoms))))
# print(harmonic_ft)


fig, axes = plt.subplots(1, 2)


calculators = [
    omat_calc,
    mace_mp_medium_reference_calc,
    ft_calc_no_replay,
    ft_calc_replay,
]

calc_names = ["OMAT", "Mace_mp", "FT-naive", "FT-replay"]

# ref_energy = np.zeros(displacement_vec.shape)
# ref_forces = np.zeros((displacement_vec.shape[0],N_atoms, 3))
#
# for idx, struct in enumerate(structures):
#    print(idx)
#    struct.calc = mace_mp_medium_reference_calc
#    ref_energy[idx] = struct.get_potential_energy()
#    ref_forces[idx,:,:]  =  struct.get_forces()
#    struct.calc = None


for calc, calc_name in zip(calculators, calc_names, strict=False):
    energy = np.zeros(displacement_vec.shape)
    forces = np.zeros((displacement_vec.shape[0], N_atoms, 3))

    eq_energy = calc.get_potential_energy(atoms)
    eq_forces = calc.get_forces(atoms)

    force_mag = np.linalg.norm(eq_forces)

    print(f"{calc_name}: Energy : {eq_energy}, force mag: {force_mag}")

    for idx, struct in enumerate(structures):
        struct.calc = calc
        energy[idx] = struct.get_potential_energy() - eq_energy
        forces[idx, :, :] = struct.get_forces()

        struct.calc = None

    force_norm = np.linalg.norm(forces, axis=(1, 2))

    axes[0].plot(displacement_vec, energy, label=calc_name)
    axes[1].plot(displacement_vec, force_norm, label=calc_name)
    axes[1].scatter(np.linalg.norm(displacement.displacement), ref_force)

plt.legend()
fig.savefig("local_pes.png")
