import numpy as np

from phonotune.materials_iterator import FileMaterialsIterator
from phonotune.phonon_data.configuration_data import ConfigFactory
from phonotune.phonon_data.data_utils import open_data
from phonotune.phonon_data.equilibrium_structure import Unitcell
from phonotune.phonon_data.phonon_data import PhononData


def test_pair_construction():
    mp_id = "mp-531340"
    # mp_id = "mp-995190"
    mp_id = "mp-10499"
    data = PhononData.load_phonon_data(mp_id)

    pc = ConfigFactory(data)
    pairs, displacement_dataset = pc.construct_all_pairs()
    print(f"{len(pairs)} pairs")

    alexandria_data = open_data(mp_id)
    uc = Unitcell.from_lattice_and_points(
        mp_id=mp_id,
        lattice=alexandria_data["unit_cell"]["lattice"],
        points=alexandria_data["unit_cell"]["points"],
        phonon_calc_supercell=alexandria_data["supercell_matrix"],
        primitive_matrix=alexandria_data.get("primitive_matrix", np.eye(3)),
    )

    phono_cell = uc.to_phonopy()
    phono_cell.generate_displacements(is_plusminus=True)
    _ = phono_cell.supercells_with_displacements
    print(phono_cell.dataset)

    for ds, ph in zip(
        phono_cell.dataset["first_atoms"],
        displacement_dataset["first_atoms"],
        strict=False,
    ):
        print(ds)
        print(f"{ph["number"]} : {ph["displacement"]}")


test_pair_construction()


def test_phonon_calculation():
    # mp_id = "mp-530748"

    for mp_id in FileMaterialsIterator(
        "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/spinel_mpids"
    ):
        data = PhononData.load_phonon_data(mp_id)
        print(mp_id)
        alexandria_data = open_data(mp_id)

        uc = Unitcell.from_lattice_and_points(
            mp_id=mp_id,
            lattice=alexandria_data["unit_cell"]["lattice"],
            points=alexandria_data["unit_cell"]["points"],
            phonon_calc_supercell=alexandria_data["supercell_matrix"],
            primitive_matrix=np.eye(3),
        )

        # (alexandria_data["supercell_matrix"])
        # alexandria_data.get(
        #    "primitive_matrix", np.eye(3)
        # ))
        # print(alexandria_data.get("primitive_matrix", np.eye(3)))
        phonon = uc.to_phonopy()

        _, displacement_dataset = ConfigFactory.construct_configs_per_phonon_data(data)

        print(len(displacement_dataset["first_atoms"]))
        phonon.dataset = displacement_dataset

        phonon.produce_force_constants()
        phonon.symmetrize_force_constants()

        data_fcs = np.array(alexandria_data["force_constants"]["elements"]).reshape(
            phonon.force_constants.shape
        )

        print(data_fcs.shape)
        print(phonon.force_constants.shape)

        print(np.sum(np.abs(data_fcs)))
        print(np.sum(np.abs(phonon.force_constants)))

        assert np.allclose(data_fcs, phonon.force_constants, atol=0.001)


test_phonon_calculation()
