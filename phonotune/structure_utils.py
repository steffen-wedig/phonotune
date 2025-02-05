import matplotlib.pyplot as plt
import yaml
from ase import Atoms
from ase.calculators import calculator
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms
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


def local_relaxation(
    atoms: Atoms,
    calculator: calculator,
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


if __name__ == "__main__":
    ase_atoms = get_low_T_Ru2Sn3_structure()
    fig = plot_low_T(ase_atoms)
    fig.savefig("./structures/low_T.png")
