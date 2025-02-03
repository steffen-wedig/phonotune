from ase import Atoms, units
from ase.build import bulk, make_supercell
from ase.calculators.cp2k import CP2K
from ase.visualize import view

from phonotune.structure_utils import get_low_T_structure, to_ase

carbon = bulk("C", "fcc", a=3.57, cubic=True)
# carbon = make_supercell(carbon, P = [[ 2 ,0, 0],[0,2,0],[0,0,2]])
view(carbon)


for cutoff in [300, 400, 500, 600, 700, 800]:
    rel_cutoff = 60
    inp = f"""&FORCE_EVAL
            &DFT
                &MGRID
                NGRIDS 4
                REL_CUTOFF {rel_cutoff}
                &END MGRID
            &END DFT
            &END FORCE_EVAL"""

    with CP2K(cutoff=cutoff * units.Rydberg, inp=inp, atoms=carbon) as calc:
        print(cutoff)

        e = carbon.get_potential_energy()
        print("Energy", e)

cutoff = 500

for rel_cutoff in [20, 40, 60, 80, 100]:
    inp = f"""&FORCE_EVAL
            &DFT
                &MGRID
                NGRIDS 4
                REL_CUTOFF {rel_cutoff}
                &END MGRID
            &END DFT
            &END FORCE_EVAL"""

    with CP2K(cutoff=cutoff * units.Rydberg, inp=inp, atoms=carbon) as calc:
        print(rel_cutoff)

        e = carbon.get_potential_energy()
        print("Energy", e)


print("cutoff 500 Ry , rel cutoff 60")
