from ase.symbols import symbols2numbers
from mace.data.atomic_data import AtomicData
from mace.tools import AtomicNumberTable, get_atomic_number_table_from_zs

from phonotune.alexandria.pair_constructor import ConfigurationPairs, PairConstructor
from phonotune.alexandria.phonon_data import PhononData


class PairDataset:
    def __init__(self, data: list[tuple[AtomicData, AtomicData]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def from_configurations(
        cls, configs: ConfigurationPairs, z_table: AtomicNumberTable, cutoff: float
    ):
        data = []

        for config0, config1 in configs:
            atomic_data0 = AtomicData.from_config(
                config0, z_table=z_table, cutoff=cutoff
            )
            atomic_data1 = AtomicData.from_config(
                config1, z_table=z_table, cutoff=cutoff
            )

            data.append((atomic_data0, atomic_data1))

        return cls(data=data)


def main():
    mp_id = "mp-556756"
    data = PhononData.load_phonon_data(
        mp_id
    )  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
    pc = PairConstructor(
        data
    )  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
    pairs = pc.construct_all_pairs()

    z_table = get_atomic_number_table_from_zs(
        symbols2numbers(data.supercell.atom_symbols)
    )  # TODO: Make this nicer

    cutoff = 5.0
    pair_dataset = PairDataset.from_configurations(
        configs=pairs, z_table=z_table, cutoff=cutoff
    )

    from mace.tools.torch_geometric.dataloader import DataLoader

    dl = DataLoader(
        pair_dataset, batch_size=10
    )  # This dataloader should return a pair of batches.

    for batch in dl:
        print(batch[0], batch[1])


if __name__ == "__main__":
    main()
