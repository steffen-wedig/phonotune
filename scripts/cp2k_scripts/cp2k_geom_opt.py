from ase import Atoms, units
from ase.calculators.cp2k import CP2K
from ase.optimize import BFGS
from ase.visualize import view

from phonotune.structure_utils import *

struct = get_low_T_structure()
atoms = to_ase(struct)


inp = """&FORCE_EVAL
        &DFT
            &MGRID
            NGRIDS 4
            REL_CUTOFF 60
            &END MGRID
        &END DFT
        &END FORCE_EVAL"""

with CP2K(cutoff=500 * units.Rydberg, inp=inp) as calc:
    atoms.calc = calc
    opt = BFGS(atoms)
    opt.run(fmax=0.05)

view(atoms)
