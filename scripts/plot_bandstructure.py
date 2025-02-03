import matplotlib.pyplot as plt
import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

path = [
    [[0, 0, 0], [0.5, 0, 0.5], [0.625, 0.25, 0.625]],
    [[0.375, 0.375, 0.75], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.25, 0.75]],
]
labels = ["$\\Gamma$", "X", "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon = phonopy.load("phonopy_params.yaml")
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
fig = phonon.plot_band_structure()

fig.set_size_inches(6, 6)
fig.suptitle("Phonon band structure of low T Mn4Si7 w MACE-MP")
fig.savefig("Mn4Si7_phononstructure.png")

fig.show()
# To plot DOS next to band structure
phonon.run_mesh([20, 20, 20])
phonon.run_total_dos()
phonon.plot_band_structure_and_dos().show()
