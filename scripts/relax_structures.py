from ase.io import write

from phonotune.calculation_configuration import (
    CalculationConfig,
    CalculationSetup,
    Material,
)

mat1 = Material(name="Mn4Si7", temperature="high")
mat2 = Material(name="Mn4Si7", temperature="low")
mat3 = Material(name="Ru2Sn3", temperature="high")
mat4 = Material(name="Ru2Sn3", temperature="low")

mats = [mat1, mat2, mat3, mat4]

for mat in mats:
    config = CalculationConfig(
        calculator_type="MACE_OMAT",
        material=mat,
        relaxation_tolerance=0.001,
        symmetry_tolerance=1e-3,
    )

    setup = CalculationSetup(config)
    calculator = setup.get_calulator()
    setup.get_atoms()
    atoms = setup.relax_atoms()

    write(
        f"{config.material.name}_{config.material.temperature}_{config.calculator_type}.xyz",
        atoms,
        format="extxyz",
    )
