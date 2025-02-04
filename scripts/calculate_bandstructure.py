import phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

path = [[[0, 0, 0], [0.5, 0, 0.5]]]
labels = ["$\\Gamma$", "X"]  # , "U", "K", "$\\Gamma$", "L", "W"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon = phonopy.load("phonopy_params.yaml")
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
phonon._band_structure.write_yaml(filename="test")
