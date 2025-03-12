from mace.calculators import mace_mp

from phonotune.alexandria.configuration_data import (
    ConfigFactory,
    ConfigSequenceDataset,
    Configuration,
)
from phonotune.alexandria.phonon_data import PhononData
from phonotune.evaluation.training_evaluation import evaluate_model_on_config_set

type ConfigurationPairs = list[tuple[Configuration, Configuration]]


def main():
    device = "cuda"
    mp_id = "mp-25"
    data = PhononData.load_phonon_data(
        mp_id
    )  # Load the Phonon Data, which reutrns a list of displacements and equilibirum structures
    pc = ConfigFactory(
        data
    )  # This converts a list of single-atom displacements into a tuple of configuration pairs. The pairs of configurations
    pairs, _ = pc.construct_all_pairs()
    print(f"{len(pairs)} pairs")

    config_seq = ConfigSequenceDataset(pairs)

    (_, valid_seq) = config_seq.train_validation_split(0.8)

    unrolled_valid_split = valid_seq.unroll()

    calc = mace_mp("small", device)

    evaluate_model_on_config_set(calc, unrolled_valid_split)


if __name__ == "__main__":
    main()
