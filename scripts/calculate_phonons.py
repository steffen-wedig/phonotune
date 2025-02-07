import phonopy
from ase.io import write
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from phonotune.calculation_configuration import (
    CalculationConfig,
    CalculationSetup,
    Material,
)
from phonotune.helper_functions import aseatoms2phonopy
from phonotune.phonon_calculations import calculate_fc2_phonopy_set

mat = Material(name="Mn4Si7", temperature="low")
config = CalculationConfig(
    calculator_type="MACE_OMAT", material=mat, relaxation_tolerance=0.001
)


mat1 = Material(name="Mn4Si7", temperature="high")
mat2 = Material(name="Mn4Si7", temperature="low")
mat3 = Material(name="Ru2Sn3", temperature="high")
mat4 = Material(name="Ru2Sn3", temperature="low")

mats = [mat1, mat2, mat3, mat4]

for clc in ["MACE_OMAT", "MACE_MP_0"]:
    for mat in mats:
        config = CalculationConfig(
            calculator_type=clc,
            material=mat,
            relaxation_tolerance=0.001,
            symmetry_tolerance=1e-3,
        )

        runname = f"{config.material.name}_{config.material.temperature}_{config.calculator_type}"

        setup = CalculationSetup(config)
        calculator = setup.get_calulator()
        setup.get_atoms()

        try:
            atoms = setup.relax_atoms()
        except AssertionError as e:
            print(e)
            continue
        else:
            write(
                f"structures/{runname}.xyz",
                atoms,
                format="extxyz",
            )

            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            phonon = aseatoms2phonopy(atoms, supercell_matrix)
            phonon.generate_displacements(distance=0.03)
            supercells = phonon.supercells_with_displacements
            force_set = calculate_fc2_phonopy_set(phonon, calculator)
            phonon.save(filename=f"{runname}_phonons.yaml")

            phonon = phonopy.load(f"{runname}_phonons.yaml")
            path = [[[0, 0, 0], [0.5, 0, 0.5]]]
            labels = ["$\\Gamma$", "X"]  # , "U", "K", "$\\Gamma$", "L", "W"]
            qpoints, connections = get_band_qpoints_and_path_connections(
                path, npoints=51
            )
            phonon.run_band_structure(
                qpoints, path_connections=connections, labels=labels
            )
            phonon._band_structure.write_yaml(filename=f"{runname}_bands.yaml")
