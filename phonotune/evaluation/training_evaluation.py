import glob
import json
from collections.abc import Sequence
from typing import Literal

import numpy as np
from mace.calculators import MACECalculator

from phonotune.alexandria.configuration_data import ConfigSingleDataset
from phonotune.evaluation.eval_utils import get_model_name_epoch_from_checkpoint_path
from phonotune.evaluation.phonon_benchmark import (
    PhononBenchmark,
)
from phonotune.model_utils import update_weights_from_checkpoint
from phonotune.structure_utils import convert_configuration_to_ase

type TrainingType = Literal["finetune_replay", "finetune_noreplay", "scratch"]


class ModelTrainingRun:
    def __init__(self, model_name, directory, training_type: TrainingType = None):
        self.model_name: str = model_name
        self.directory: str = directory
        self.trainig_type: TrainingType | None = training_type
        self.max_epochs = None
        self.checkpoint_paths: dict[int:str] = self.get_checkpoints()

        self.loss_txt_file: str | None = glob.glob(f"{self.directory}/results/*")[0]
        self.model_file = self.get_model_filepath()
        self.N_samples = self.get_N_samples()
        self.validation_loss = Sequence[float]

    def get_model_filepath(self):
        paths = glob.glob(
            f"{self.directory}/*.model"
        )  # returns the path of the compiled and the regular, uncompiled model

        for model_path in paths:
            if "compiled" not in model_path:
                print(model_path)
                return model_path

    def get_training_type(self):
        if "finetune" in self.model_name:
            if "replay" in self.model_name:
                self.trainig_type = "finetune_replay"
            else:
                self.trainig_type = "finetune_noreplay"
        else:
            self.trainig_type = "scratch"

        return self.trainig_type

    def get_N_samples(self):
        self.N_samples = int(self.model_name.split("_")[-1])

    def get_validation_loss(self):
        if self.trainig_type == "finetune_replay":
            self.validation_loss, self.replay_validation_loss = (
                self.load_loss_values_replay(self.loss_txt_file)
            )
        else:
            self.validation_loss = self.load_loss_values(self.loss_txt_file)

    def get_checkpoints(self):
        checkpoint_paths = glob.glob(f"{self.directory}/checkpoints/*.pt")

        names = []
        epochs = []
        for path in checkpoint_paths:
            name, epoch = get_model_name_epoch_from_checkpoint_path(path)
            names.append(name)
            epochs.append(epoch)

        assert len(set(names)) == 1

        sort_idx = np.argsort(np.array(epochs, dtype=int)).tolist()
        epochs = sorted(epochs)
        checkpoint_paths = [checkpoint_paths[i] for i in sort_idx]
        checkpoint_paths = {
            epoch: path for epoch, path in zip(epochs, checkpoint_paths, strict=False)
        }

        self.max_epochs = epochs[-1]
        return checkpoint_paths

    @staticmethod
    def load_loss_values(result_txt_file: str):
        validation_loss = {}
        with open(result_txt_file) as f:
            for line in f.readlines():
                data = json.loads(line)

                if data["mode"] == "eval":
                    if data["epoch"] is None:
                        epoch = 0
                    else:
                        epoch = int(data["epoch"]) + 1

                    validation_loss[epoch] = float(data["rmse_f"]) * 1000

        return validation_loss

    @staticmethod
    def load_loss_values_replay(result_txt_file: str):
        replay_validation_loss = {}
        validation_loss = {}

        with open(result_txt_file) as f:
            for line in f.readlines():
                data = json.loads(line)

                if data["mode"] == "eval":
                    if data["epoch"] is None:
                        epoch = 0
                    else:
                        epoch = int(data["epoch"]) + 1

                    if epoch not in replay_validation_loss:
                        # The replay validation losses are listed first, so we have to check whether they already exists or no
                        replay_validation_loss[epoch] = float(data["rmse_f"]) * 1000
                    else:
                        validation_loss[epoch] = float(data["rmse_f"]) * 1000

        return validation_loss, replay_validation_loss

    def evaluate_checkpoints_on_configs(
        self, test_dataset: ConfigSingleDataset, evaluation_interval: int
    ):
        rmses = {}
        for epoch, path in self.checkpoint_paths.items():
            if epoch % evaluation_interval == 0 or epoch == self.max_epochs:
                mace_calc = MACECalculator(
                    model_paths=self.model_file,
                    device="cuda",
                    enable_cueq=True,
                )

                mace_calc = update_weights_from_checkpoint(
                    mace_calculator=mace_calc, checkpoint_path=path
                )

                rmse_epoch = self.evaluate_model_on_config_set(mace_calc, test_dataset)

                print(f"Epoch {epoch}, RMSE: {rmse_epoch}")
                rmses[epoch + 1] = (
                    rmse_epoch  # Add the 1 to ensure consistency with the validation loss plot. Here we one index the epochs, instead of 0 index as in the trainig log
                )

        return rmses

    @staticmethod
    def evaluate_model_on_config_set(
        mace_calc: MACECalculator, config_sequence: ConfigSingleDataset
    ):
        rmses = []

        for configuration in config_sequence.data:
            ase_atoms = convert_configuration_to_ase(configuration)
            ase_atoms.calc = mace_calc
            mace_forces = ase_atoms.get_forces()
            dft_forces = ase_atoms.get_array("DFT_forces")
            rmse_forces = np.sqrt(np.mean((mace_forces - dft_forces) ** 2)) * 1000
            rmses.append(rmse_forces)

        return np.mean(np.array(rmses)).item()

    def evaluate_model_on_phonons(self, test_mp_ids):
        mace_calc = MACECalculator(
            model_paths=self.model_file,
            head="default",
            device="cuda",
            enable_cueq=True,
        )

        comp = PhononBenchmark.construct_from_mpids(mace_calc, test_mp_ids)

        td_mae_dict = comp.calculate_thermodynamic_MAE()
        phonon_mse, freq, errors = comp.compare_datasets_phonons()
        print(f"Phonon MSE {phonon_mse}")
        filepath = f"{self.directory}/test_mp_ids_phonon_comp_{self.model_name}.yaml"
        comp.store_data(filepath, calculator_name=self.model_name)

        self.phonon_evaluation = comp

        return td_mae_dict, phonon_mse
