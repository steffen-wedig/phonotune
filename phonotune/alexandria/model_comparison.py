import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure

from phonotune.alexandria.data_utils import serialization_dict_type_conversion
from phonotune.alexandria.phonon_data import PhononData, PhononDataset


class ModelComparison:
    def __init__(self, dataset_ref: PhononDataset, dataset_pred: PhononDataset):
        self.dataset_ref = dataset_ref
        self.dataset_pred = dataset_pred
        self.td_deltas = None
        self.phonon_errors = None

    @staticmethod
    def compare_phonon_td_data(
        phonon_data_ref: PhononData, phonon_data_pred: PhononData
    ):
        assert phonon_data_pred.mp_id == phonon_data_ref.mp_id

        index_300_K = np.argwhere(
            phonon_data_pred.properties.temperatures == 300.0
        ).item(0)

        N_atoms = phonon_data_ref.properties.N_atoms_unitcell

        energy_delta = (
            phonon_data_ref.properties.energy - phonon_data_pred.properties.energy
        ) / N_atoms
        volume_delta = (
            phonon_data_ref.properties.volume - phonon_data_pred.properties.volume
        ) / N_atoms
        print(f"Ref Vol {phonon_data_ref.properties.volume}")
        print(f"Pred Vol {phonon_data_pred.properties.volume}")

        entropy_delta = (
            phonon_data_ref.properties.entropy[index_300_K]
            - phonon_data_pred.properties.entropy[index_300_K]
        )
        free_energy_delta = (
            phonon_data_ref.properties.free_energy[index_300_K]
            - phonon_data_pred.properties.free_energy[index_300_K]
        )
        heat_capacity_delta = (
            phonon_data_ref.properties.heat_capacity[index_300_K]
            - phonon_data_pred.properties.heat_capacity[index_300_K]
        )

        return (
            energy_delta,
            volume_delta,
            entropy_delta,
            free_energy_delta,
            heat_capacity_delta,
        )

    def store_data(self, filepath: str, calculator_name=None):
        if calculator_name is None:
            calculator_name = self.dataset_pred.get_source()

        assert self.td_deltas is not None
        assert self.phonon_errors is not None
        assert self.mae_dict is not None

        data = {
            "mace_model": calculator_name,
            "mp_ids": self.dataset_pred.get_mp_ids(),
            "td_deltas": self.td_deltas,
            "phonon_error": self.phonon_errors,
            "td_maes": self.mae_dict,
        }

        with open(filepath, "w") as f:
            yaml.dump(serialization_dict_type_conversion(data), f)

    def compare_datasets_td(self):
        N_samples = len(self.dataset_ref.phonon_data_samples)

        energy_deltas = np.zeros(shape=N_samples)
        volume_deltas = np.zeros(shape=N_samples)
        entropy_deltas = np.zeros(shape=N_samples)
        free_energy_deltas = np.zeros(shape=N_samples)
        heat_capacity_deltas = np.zeros(shape=N_samples)

        for i, (phonon_data_ref, phonon_data_pred) in enumerate(
            zip(
                self.dataset_ref.phonon_data_samples,
                self.dataset_pred.phonon_data_samples,
                strict=False,
            )
        ):
            (
                energy_deltas[i],
                volume_deltas[i],
                entropy_deltas[i],
                free_energy_deltas[i],
                heat_capacity_deltas[i],
            ) = self.compare_phonon_td_data(phonon_data_ref, phonon_data_pred)

        self.td_deltas = {
            "energy": energy_deltas,
            "volume": volume_deltas,
            "entropy": entropy_deltas,
            "free_energy": free_energy_deltas,
            "heat_capacity": heat_capacity_deltas,
        }

        return self.td_deltas

    def calculate_MAE(self):
        if self.td_deltas is None:
            _ = self.compare_datasets_td()

        mae_dict = {}

        for key in self.td_deltas.keys():
            mae_dict[key] = np.mean(np.abs(self.td_deltas[key])).item()

        self.mae_dict = mae_dict

        return mae_dict

    def compare_datasets_phonons(self):
        ref_frequencies = None
        errors = None
        for phonon_data_ref, phonon_data_pred in zip(
            self.dataset_ref.phonon_data_samples,
            self.dataset_pred.phonon_data_samples,
            strict=False,
        ):
            sample_ref_freq = np.array(phonon_data_ref.phonon_spectrum["frequencies"])

            sample_pred_freq = np.array(phonon_data_pred.phonon_spectrum["frequencies"])

            frequency_error = (sample_ref_freq - sample_pred_freq) ** 2

            if ref_frequencies is None:
                # For the first sample
                ref_frequencies = sample_ref_freq.reshape(
                    -1,
                )
                errors = frequency_error.reshape(
                    -1,
                )

            else:
                ref_frequencies = np.hstack(
                    (
                        ref_frequencies,
                        sample_ref_freq.reshape(
                            -1,
                        ),
                    )
                )
                errors = np.hstack(
                    (
                        errors,
                        frequency_error.reshape(
                            -1,
                        ),
                    )
                )

        self.phonon_errors = {"ref_freq": ref_frequencies, "errors": errors}

        mse = np.mean(errors)

        return mse, ref_frequencies, errors


class Visualizer:
    def __init__(self, *model_names, N_materials):
        self.models = model_names
        self.N_materials = N_materials
        self.phonon_data_dicts = self.load_model_data()

    def load_model_data(self):
        phonon_data_dicts = []

        for model in self.models:
            data_dict = yaml.safe_load(
                open(
                    f"/data/fast-pc-06/snw30/projects/phonons/phonotune/data/phonon_comp_{self.N_materials}_{model}.yaml"
                )
            )
            phonon_data_dicts.append(data_dict)

        return phonon_data_dicts

    def print_td_maes(self):
        for i, model in enumerate(self.models):
            assert model == self.phonon_data_dicts[i]["mace_model"]
            maes = self.phonon_data_dicts[i]["td_maes"]
            print(model)
            print(maes)

    def make_violin_plots(self) -> Figure:
        errors = self.get_error_arrays()

        N_models = errors.shape[0]
        N_categories = errors.shape[1]

        category_names = self.get_categories()

        # Create one subplot per category. Sharing the y-axis for consistency.
        fig, axs = plt.subplots(
            1, N_categories, figsize=(4 * N_categories, 6), sharey=True
        )

        # If there's only one category, ensure axs is iterable.
        if N_categories == 1:
            axs = [axs]

        # Define a colormap for distinct model colors.
        cmap = plt.get_cmap("Set2")
        model_colors = [cmap(i) for i in range(N_models)]

        # Plot each category in its own axis.
        for j in range(N_categories):
            ax = axs[j]
            # Define x positions for the models (e.g., 1, 2, 3, …)
            positions = np.arange(1, N_models + 1)
            for m in range(N_models):
                data = errors[m, j, :].squeeze()
                vp = ax.violinplot(
                    data,
                    positions=[positions[m]],
                    widths=0.2,
                    quantiles=[0, 1],
                    showmeans=True,
                    orientation="horizontal",
                )
                for body in vp["bodies"]:
                    body.set_facecolor(model_colors[m])
                    body.set_edgecolor("black")
                    body.set_alpha(0.7)
            ax.set_title(category_names[j])

        plt.tight_layout()

        return fig

    def get_categories(self):
        categories = self.phonon_data_dicts[0]["td_deltas"].keys()
        return list(categories)

    def get_number_materials(self):
        check_mp_ids_equivalent_set = None

        for i in self.phonon_data_dicts:
            mp_ids = i["mp_ids"]

            if check_mp_ids_equivalent_set is None:
                check_mp_ids_equivalent_set = set(mp_ids)

            else:
                assert check_mp_ids_equivalent_set == set(mp_ids)

        return len(mp_ids)

    def get_error_arrays(self):
        categories = self.get_categories()
        N_models = len(self.models)
        N_categories = len(categories)
        N_deltas = self.get_number_materials()
        errors = np.zeros(shape=(N_models, N_categories, N_deltas))
        for model_idx, _ in enumerate(self.models):
            td_deltas = self.phonon_data_dicts[model_idx]["td_deltas"]
            for cat_idx, category in enumerate(categories):
                errors[model_idx, cat_idx, :] = np.array(td_deltas[category])

        return errors
