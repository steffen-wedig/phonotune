import bz2
import json
import os

import numpy as np
import requests
import yaml

from phonotune.materials_iterator import MaterialsIterator


def download_and_unpack_phonons(mp_id):
    url = f"https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/{mp_id}.yaml.bz2"
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Decompress the file content using bz2
    decompressed_bytes = bz2.decompress(response.content)

    # Convert the decompressed bytes to a string (assuming UTF-8 encoding)
    yaml_str = decompressed_bytes.decode("utf-8")

    # Parse the YAML string into Python data structures
    data = yaml.safe_load(yaml_str)

    return data


def to_yaml(data, mp_id):
    if not os.path.isdir("data/materials"):
        os.mkdir("data/materials")
    yaml.dump(data, open(f"data/materials/{mp_id}.yaml", "w"))


def from_yaml(mp_id):
    data = yaml.safe_load(open(f"data/materials/{mp_id}.yaml"))
    return data


def download_and_unpack_relaxation_traj(traj):
    url = f"https://alexandria.icams.rub.de/data/pbe/geo_opt_paths/alex_go_{traj}.json.bz2"

    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Decompress the file content using bz2
    decompressed_bytes = bz2.decompress(response.content)

    # Convert the decompressed bytes to a string (assuming UTF-8 encoding)
    json_str = decompressed_bytes.decode("utf-8")

    # Parse the YAML string into Python data structures
    data = json.loads(json_str)

    return data


def open_data(mp_id):
    # TODO: clean the mp_id check whether it starts with mp and add that
    # type cast it to a string.

    try:
        data = from_yaml(mp_id)
    except FileNotFoundError:
        data = download_and_unpack_phonons(mp_id)
        to_yaml(data, mp_id)
    return data


def is_unstable_lattice(data):
    freq = data["phonon_freq"]
    if np.min(np.array(freq)) < -1e-3:
        return True

    return False


def contains_non_mace_elements(data):
    non_mace_elements_in_alexandria = {"Th", "Pa", "U"}
    elements = set()
    for point in data["unit_cell"]["points"]:
        elements.update(point["symbol"])

    if non_mace_elements_in_alexandria & elements:
        # Non zero intersection between non mace elements and the elements in this data point
        return True
    else:
        return False


def search_highest_number_displacements(mat_iterator: MaterialsIterator):
    max_displacements = 0
    max_displacements_mp_id = None
    while True:
        try:
            mp_id = next(mat_iterator)
            data = open_data(mp_id)
            if np.allclose(np.array(data["supercell_matrix"]), np.eye(3)):
                continue
            num_displacements = len(data["displacements"])
            if num_displacements > max_displacements:
                print(num_displacements)
                print(mp_id)
                max_displacements = num_displacements
                max_displacements_mp_id = mp_id

        except StopIteration:
            return max_displacements, max_displacements_mp_id


def check_last_atom_displaced(mat_iterator: MaterialsIterator):
    while True:
        mp_id = next(mat_iterator)
        data = open_data(mp_id)

        num_atoms = len(data["supercell"]["points"])

        max_disp_atom = data["displacements"][-1]["atom"]
        min_disp_atom = data["displacements"][0]["atom"]

        if min_disp_atom == 0:
            print(f"Zero atom at {mp_id}")

        if num_atoms == max_disp_atom:
            print("Last atom reached")
            print(mp_id)
            break


def serialization_dict_type_conversion(data_dict: dict):
    new_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            new_dict[key] = value.tolist()
        elif isinstance(value, dict):
            new_dict[key] = serialization_dict_type_conversion(value)
        elif isinstance(value, np.ndarray) and value.shape == (1,):
            new_dict[key] = value.item()
        else:
            new_dict[key] = value
    return new_dict


def unpack_points(points):
    # Unpacks the "points" dicts in the alexandria phonon dataset
    N_atoms = len(points)
    frac_coordinates = np.zeros(shape=(N_atoms, 3))
    atom_symbols = []
    for idx, p in enumerate(points):
        frac_coordinates[idx, :] = p["coordinates"]
        atom_symbols.append(p["symbol"])

    return frac_coordinates, atom_symbols


def get_displacement_dataset_from_alexandria_data_dict(data):
    displacement_data = []

    _, atom_symbols = unpack_points(data["supercell"]["points"])

    for disp in data["displacements"]:
        displacement_data.append(
            {
                "number": disp["atom"] - 1,
                "displacement": np.array(disp["displacement"], dtype=np.float64),
                "forces": np.array(disp["forces"], dtype=np.float64),
            }
        )

    displacement_dataset = {
        "natoms": len(atom_symbols),
        "first_atoms": displacement_data,
    }

    return displacement_dataset
