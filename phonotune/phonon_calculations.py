from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from phonopy.api_phonopy import Phonopy
from tqdm import tqdm

from phonotune.helper_functions import (
    aseatoms2phonopy,
    get_chemical_formula,
)

FREQ_CUTOFF = 1e-3


def calculate_forces_phonopy_set(
    phonons: Phonopy,
    calculator: Calculator,
    log: bool = True,
) -> np.ndarray:
    # calculate FC2 force set

    forces = []
    nat = len(phonons.supercell)

    for sc in tqdm(
        phonons.supercells_with_displacements,
        desc=f"FC2 calculation: {get_chemical_formula(phonons)}",
    ):
        if sc is not None:
            atoms = Atoms(sc.symbols, cell=sc.cell, positions=sc.positions, pbc=True)
            atoms.calc = calculator
            f = atoms.get_forces()
        else:
            f = np.zeros((nat, 3))
        forces.append(f)

    # append forces
    force_set = np.array(forces)
    phonons.forces = force_set
    return force_set


def init_phonopy(
    atoms: Atoms,
    fc2_supercell: np.ndarray | None = None,
    primitive_matrix: Any = "auto",
    log: str | Path | bool = True,
    symprec: float = 1e-5,
    displacement_distance: float = 0.03,
    **kwargs: Any,
) -> tuple[Phonopy, list[Any]]:
    """Calculate fc2 and fc3 force lists from phonopy."""
    if not log:
        log_level = 0
    elif log is not None:
        log_level = 1

    if fc2_supercell is not None:
        _fc2_supercell = fc2_supercell
    else:
        if "fc2_supercell" in atoms.info.keys():
            _fc2_supercell = atoms.info["fc2_supercell"]
        else:
            raise ValueError(
                f'{atoms.get_chemical_formula(mode="metal")=} "fc2_supercell" was not found in atoms.info and was not provided as an argument when calculating force sets.'
            )

    # Initialise Phonopy object
    phonons = aseatoms2phonopy(
        atoms,
        fc2_supercell=_fc2_supercell,
        primitive_matrix=primitive_matrix,
        symprec=symprec,
        log_level=log_level,
        **kwargs,
    )

    phonons.generate_displacements(distance=displacement_distance)

    return phonons


def get_fc2_and_freqs(
    phonons: Phonopy,
    calculator: Calculator | None = None,
    q_mesh: np.ndarray | None = None,
    symmetrize_fc2=True,
    log: str | Path | bool = True,
    pbar_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Phonopy, np.ndarray, np.ndarray]:
    if calculator is None:
        raise ValueError(
            f'{get_chemical_formula(phonons)} "calculator" was provided when calculating fc2 force sets.'
        )

    if not pbar_kwargs:
        pbar_kwargs = {"leave": False}

    fc2_set = calculate_forces_phonopy_set(
        phonons, calculator, log=log, pbar_kwargs=pbar_kwargs
    )

    phonons.produce_force_constants(show_drift=False)

    if symmetrize_fc2:
        phonons.symmetrize_force_constants(show_drift=False)

    if q_mesh is not None:
        phonons.run_mesh(q_mesh, **kwargs)
        freqs = phonons.get_mesh_dict()["frequencies"]
    else:
        freqs = []

    return phonons, fc2_set, freqs


def load_force_sets(phonons: Phonopy, fc2_set: np.ndarray) -> Phonopy:
    phonons.forces = fc2_set
    phonons.produce_force_constants(symmetrize_fc2=True)
    return phonons
