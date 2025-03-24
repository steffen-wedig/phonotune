from collections.abc import Iterable

import numpy as np
from ase.io import read
from elementembeddings.composition import CompositionalEmbedding
from mp_api.client import MPRester
from tqdm import tqdm
from umap import UMAP

from phonotune.materials_iterator import (
    FileMaterialsIterator,
    MaterialsIterator,
)
from phonotune.phonon_data.data_utils import get_mp_api_key


def get_chemical_formulas_from_mp_id(mat_iterator: MaterialsIterator):
    mp_id_list = list(mat_iterator)

    api_key = get_mp_api_key()

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=mp_id_list)

    formulas = []
    for doc in docs:
        formulas.append(doc.formula_pretty)

    return formulas


def get_formulas_from_replay_dataset(replay_xyz_path: str):
    materials = read(replay_xyz_path, index=":")

    formulas = []
    for mat in materials:
        formulas.append(mat.get_chemical_formula())

    return formulas


def assemble_full_formula_dataset():
    replay_data_path = (
        "/data/fast-pc-06/snw30/projects/phonons/training/mp_finetuning-replay.xyz"
    )

    alexandria_ood_path = "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/general_test_alexandria_mpids"

    training_data_path = (
        "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/spinel_mpids"
    )

    formulas_mp_replay = get_formulas_from_replay_dataset(replay_data_path)

    alexandria_ood_mp_ids = FileMaterialsIterator(alexandria_ood_path)
    formulas_alexandria_ood = get_chemical_formulas_from_mp_id(alexandria_ood_mp_ids)

    training_data_mp_ids = FileMaterialsIterator(training_data_path)
    formulas_training_data = get_chemical_formulas_from_mp_id(training_data_mp_ids)

    subsets = [formulas_mp_replay, formulas_alexandria_ood, formulas_training_data]

    names = ["Replay", "Alexandria OOD", "Aluminium Oxides"]
    return subsets, names


def train_reducer(embeddings, n_components=2):
    reducer = UMAP(n_components=n_components)
    reducer.fit_transform(embeddings)

    return reducer


# This function is taken and slightly modified from https://smact.readthedocs.io/en/latest/tutorials/crystal_space_visualisation.html
def get_embedding(formula, embedding="magpie", stats="mean"):
    if isinstance(formula, str):
        formula = [formula]
    elif isinstance(formula, Iterable):
        pass
    else:
        raise TypeError("formula must be a string or a list of strings")

    # get embedding dimension
    # compute embedding
    embeddings = []
    for f in tqdm(formula):
        try:
            compositional_embedding = CompositionalEmbedding(f, embedding=embedding)
            embeddings.append(compositional_embedding.feature_vector(stats=stats))
        except Exception:
            # the exception is raised when the embedding doesn't support the element
            continue
            # embeddings.append(np.full(embedding_dim, np.nan))

    # concatenate the embedded vectors
    embeddings = np.stack(embeddings, axis=0).squeeze()
    return embeddings


def collect_formulas(formulas_subsets: list[list[str]]):
    formulas = []

    for f in formulas_subsets:
        formulas.extend(f)

    return formulas
