import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE, LBFGS
from pymatgen.ext.matproj import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from phonotune.configuration import Configuration
from phonotune.materials_iterator import MaterialsIterator


def get_from_mp(material_id):
    yaml_file = "./api_key.yaml"
    with open(yaml_file) as f:
        api_key = yaml.safe_load(f)["MP_API_KEY"]

    with MPRester(api_key) as m:
        # Structure for material id
        structure = m.get_structure_by_material_id(material_id)
    return structure


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
    relaxation_tolerance: float = 0.0005,
    N_max_steps=10000,
):
    atoms.calc = calculator
    # atoms.set_constraint(FixSymmetry(atoms))
    sym_filter = FrechetCellFilter(atoms)
    opt = FIRE(sym_filter, logfile="/dev/null")
    converged = opt.run(fmax=relaxation_tolerance, steps=N_max_steps)

    if not converged:
        print(f"Not Converged in {N_max_steps} steps")
        raise ValueError

    return atoms


def unitcell_fire_relaxation(
    unitcell, calculator: Calculator, relaxation_tolerance=0.005, N_max_steps=10000
):
    atoms: Atoms = unitcell.to_ase_atoms()
    try:
        atoms = local_fire_relaxation(
            atoms, calculator, relaxation_tolerance, N_max_steps
        )
        # Overwrite the field in the unitcell
        unitcell.fractional_coordinates = atoms.get_scaled_positions()
        unitcell.lattice = atoms.get_cell()

        print(f"Vol: {atoms.get_volume()}")
    except ValueError:
        raise

    return unitcell


def convert_configuration_to_ase(configuration: Configuration) -> Atoms:
    atoms = Atoms(
        numbers=configuration.atomic_numbers,
        positions=configuration.positions,
        cell=configuration.cell,
        pbc=configuration.pbc,
    )

    atoms.set_array("DFT_forces", configuration.properties["DFT_forces"])
    # atoms.info["MACE_energy"] = 0.0
    return atoms


def get_spinel_group_mpids(mp_ids: MaterialsIterator):
    yaml_file = "./api_key.yaml"

    with open(yaml_file) as f:
        api_key = yaml.safe_load(f)["MP_API_KEY"]

    spinel_list = []
    # Fetch the reference spinel structure

    with MPRester(api_key) as mpr:
        docs_al = mpr.materials.summary.search(
            chemsys="*-Al-O", formula="AB2C4", fields=["material_id"]
        )

        docs_bin = mpr.materials.summary.search(
            chemsys="Al-O", formula="A3B4", fields=["material_id"]
        )

        docs_al.extend(docs_bin)
        docs = [doc.material_id for doc in docs_al]

        search_space = set(mp_ids) & set(docs)

        spinel_list = list(search_space)

        docs = mpr.materials.summary.search(
            material_ids=spinel_list, fields=["formula_pretty", "elements"]
        )

        formulas = [i.formula_pretty for i in docs]

        elements = set()

        for i in docs:
            elements.update(i.elements)

        return spinel_list, formulas, elements
