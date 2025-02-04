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

setup = CalculationSetup(config)
calculator = setup.get_calulator()
atoms = setup.get_atoms()

supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
phonon = aseatoms2phonopy(atoms, supercell_matrix)
phonon.generate_displacements(distance=0.03)
supercells = phonon.supercells_with_displacements
force_set = calculate_fc2_phonopy_set(phonon, calculator)
phonon.save()
