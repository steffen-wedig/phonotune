import glob
import os

import matplotlib.pyplot as plt

from phonotune.alexandria.configuration_data import ConfigSingleDataset
from phonotune.evaluation.plotting_utils import (
    plot_val_loss_over_N_samples,
    plot_validation_loss_curves_over_epoch,
)
from phonotune.evaluation.training_evaluation import ModelTrainingRun

# Get all entries matching the pattern, e.g. all items in the current directory
TRAINING_DIR = "/data/fast-pc-06/snw30/projects/phonons/training"
entries = glob.glob(f"{TRAINING_DIR}/*")

# Filter out only directories
training_runs = [
    os.path.basename(os.path.normpath(d)) for d in entries if os.path.isdir(d)
]


noreplay_runs = []
replay_runs = []
for training_run in training_runs:
    tr = ModelTrainingRun(training_run, f"{TRAINING_DIR}/{training_run}")
    tr.get_training_type()
    tr.get_N_samples()
    tr.get_validation_loss()
    if tr.trainig_type == "finetune_noreplay":
        noreplay_runs.append(tr)
    elif tr.trainig_type == "finetune_replay":
        replay_runs.append(tr)

noreplay_runs.sort(key=lambda x: x.N_samples)
replay_runs.sort(key=lambda x: x.N_samples)


fig = plot_validation_loss_curves_over_epoch(noreplay_runs, replay_runs)
plt.savefig("model_val_loss.svg", dpi=300)


min_val_loss_by_N_samples_replay = {}
min_val_loss_by_N_samples_noreplay = {}

for tr in noreplay_runs:
    min_val_loss_by_N_samples_noreplay[tr.N_samples] = min(tr.validation_loss.values())

for tr in replay_runs:
    min_val_loss_by_N_samples_replay[tr.N_samples] = min(tr.validation_loss.values())


fig = plot_val_loss_over_N_samples(
    min_val_loss_by_N_samples_replay, min_val_loss_by_N_samples_noreplay
)
plt.savefig("loglogsample_val_loss.svg", dpi=300)

generalization_test_config_dataset = ConfigSingleDataset.from_hdf5(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/general_test_configs.h5"
)


noreplay_run_1000 = noreplay_runs[3]

rmses = noreplay_run_1000.evaluate_checkpoints_on_configs(
    generalization_test_config_dataset
)

replay_run_1000 = replay_runs[3]

replay_rmses = replay_run_1000.evaluate_checkpoints_on_configs(
    generalization_test_config_dataset
)

print(replay_rmses)
print(rmses)
