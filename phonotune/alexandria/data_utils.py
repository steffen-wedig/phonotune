import bz2
import json
import os

import requests
import yaml


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
    if not os.path.isdir("data"):
        os.mkdir("data")
    yaml.dump(data, open(f"data/{mp_id}.yaml", "w"))


def from_yaml(mp_id):
    data = yaml.safe_load(open(f"data/{mp_id}.yaml"))
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
