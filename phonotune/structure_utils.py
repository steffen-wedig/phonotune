from collections.abc import Sequence

import matplotlib.pyplot as plt
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io.extxyz import write_extxyz
from ase.optimize import FIRE, LBFGS
from ase.visualize.plot import plot_atoms
from mace.data.atomic_data import Configuration
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor


def get_low_T_Ru2Sn3_structure():
    a = 12.344
    b = 9.922
    c = 6.161

    alpha = 90
    beta = 90
    gamma = 90
    coords = [
        [1, 0.5696, 0.75],
        [0.7491, 0.3121, 0.2424],
        [0.5, 0.4466, 0.25],
        [0.6608, 0.5760, 0.0794],
        [0.8573, 0.5909, 0.4094],
        [0.5766, 0.7351, 0.4828],
    ]

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    struct = Structure.from_spacegroup(
        "Pbcn", lattice, ["Ru", "Ru", "Ru", "Sn", "Sn", "Sn"], coords
    )

    return struct


def get_from_mp(material_id):
    yaml_file = "./api_key.yaml"
    with open(yaml_file) as f:
        api_key = yaml.safe_load(f)["MP_API_KEY"]

    with MPRester(api_key) as m:
        # Structure for material id
        structure = m.get_structure_by_material_id(material_id)
    return structure


def plot_low_T(struct):
    ase_atoms = to_ase(struct)
    fig, ax = plt.subplots()
    plot_atoms(ase_atoms, ax, radii=0.5, rotation=("90x,0y,0z"))
    return fig


def to_ase(struct) -> Atoms:
    ase_atoms = AseAtomsAdaptor().get_atoms(struct)
    return ase_atoms


def local_lbfgs_relaxation(
    atoms: Atoms,
    calculator: Calculator,
    rattle: float | None = None,
    relaxation_tolerance: float = 0.01,
):
    atoms.calc = calculator
    # also relax the cell shape?
    if rattle is not None:
        atoms.rattle(stdev=rattle)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=relaxation_tolerance)

    return atoms


def local_fire_relaxation(
    atoms: Atoms,
    calculator: Calculator,
    relaxation_tolerance: float = 0.005,
    N_max_steps=1000,
):
    atoms.calc = calculator
    atoms.set_constraint(FixSymmetry(atoms))
    sym_filter = FrechetCellFilter(atoms)
    opt = FIRE(sym_filter, logfile="/dev/null")
    converged = opt.run(fmax=relaxation_tolerance, steps=N_max_steps)

    if not converged:
        print(f"Not Converged in {N_max_steps} steps")
        raise ValueError


def unitcell_fire_relaxation(
    unitcell, calculator: Calculator, relaxation_tolerance=0.005, N_max_steps=1000
):
    atoms: Atoms = unitcell.to_ase_atoms()
    try:
        local_fire_relaxation(atoms, calculator, relaxation_tolerance, N_max_steps)
        # Overwrite the field in the unitcell
        unitcell.fractional_coordinates = atoms.get_scaled_positions()
        unitcell.lattice = atoms.get_cell()
    except ValueError:
        raise

    return unitcell


def convert_configuration_to_ase(configuration: Configuration, calc) -> Atoms:
    atoms = Atoms(
        numbers=configuration.atomic_numbers,
        positions=configuration.positions,
        cell=configuration.cell,
        pbc=configuration.pbc,
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    atoms.calc = None

    atoms.set_array("DFT_forces", configuration.properties["forces"])
    atoms.info["MACE_energy"] = energy
    return atoms


def configurations_to_xyz(xyz_file_path: str, configs: Sequence[Configuration], calc):
    ase_atoms = []

    for config in configs:
        atoms = convert_configuration_to_ase(config, calc)
        ase_atoms.append(atoms)

    f = open(xyz_file_path, "w")
    write_extxyz(
        f, ase_atoms, columns=["symbols", "positions", "DFT_forces"], write_info=True
    )
