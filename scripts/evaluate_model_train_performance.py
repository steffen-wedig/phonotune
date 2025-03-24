import glob
import os
import warnings

import matplotlib.pyplot as plt
import yaml
from mace.calculators import MACECalculator, mace_mp

from phonotune.evaluation.phonon_benchmark import PhononBenchmark
from phonotune.evaluation.plotting_utils import (
    plot_forgetting_loss,
    plot_forgetting_loss_weight_decay,
    plot_thermodynamic_property_errors,
    plot_val_loss_over_N_samples,
    plot_validation_loss_curves_over_epoch,
)
from phonotune.evaluation.training_evaluation import ModelTrainingRun
from phonotune.materials_iterator import FileMaterialsIterator
from phonotune.phonon_calculation.plotting_bands import (
    plot_model_reference_phonon_comparison,
)
from phonotune.phonon_data.configuration_data import (
    ConfigFactory,
    ConfigSingleDataset,
)
from phonotune.phonon_data.equilibrium_structure import Unitcell
from phonotune.phonon_data.phonon_data import PhononData, PhononDataset
from phonotune.structure_utils import unitcell_fire_relaxation

warnings.filterwarnings("ignore", category=FutureWarning)

# Get all entries matching the pattern, e.g. all items in the current directory
TRAINING_DIR = "/data/fast-pc-06/snw30/projects/phonons/training"
entries = glob.glob(f"{TRAINING_DIR}/*")

mace_mp_calc = mace_mp(
    "medium", device="cuda", enable_cueq=True, default_dtype="float64"
)

omat_model_file = "/data/fast-pc-06/snw30/projects/models/mace-omat-0-medium.model"
omat_calc = MACECalculator(omat_model_file, device="cuda", enable_cueq=True)


# Filter out only directories
training_runs = [
    os.path.basename(os.path.normpath(d)) for d in entries if os.path.isdir(d)
]

print(training_runs)

type TrainingRuns = list[ModelTrainingRun]

noreplay_runs: TrainingRuns = []
replay_runs: TrainingRuns = []
weight_decay_runs: TrainingRuns = []

for training_run in training_runs:
    tr = ModelTrainingRun(training_run, f"{TRAINING_DIR}/{training_run}")
    tr.get_training_type()
    tr.get_N_samples()
    tr.get_validation_loss()
    if tr.trainig_type == "finetune_noreplay":
        noreplay_runs.append(tr)
    elif tr.trainig_type == "finetune_replay":
        replay_runs.append(tr)

    elif tr.trainig_type == "weight_decay":
        weight_decay_runs.append(tr)


noreplay_runs.sort(key=lambda x: x.N_samples)
replay_runs.sort(key=lambda x: x.N_samples)
weight_decay_runs.sort(key=lambda x: x.N_samples)

# ============================================================================
# Training Curves


fig = plot_validation_loss_curves_over_epoch(
    noreplay_runs, replay_runs, weight_decay_runs
)

plt.savefig("model_val_loss.pdf", dpi=300, bbox_inches="tight")


min_val_loss_by_N_samples_replay = {}
min_val_loss_by_N_samples_noreplay = {}

for tr in noreplay_runs:
    min_val_loss_by_N_samples_noreplay[tr.N_samples] = min(tr.validation_loss.values())

for tr in replay_runs:
    min_val_loss_by_N_samples_replay[tr.N_samples] = min(tr.validation_loss.values())

fig = plot_val_loss_over_N_samples(
    min_val_loss_by_N_samples_noreplay, min_val_loss_by_N_samples_replay
)
plt.savefig("loglogsample_val_loss.pdf", dpi=300, bbox_inches="tight")


# ============================================================================
# FORGETTING


generalization_test_config_dataset = ConfigSingleDataset.from_hdf5(
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/general_test_configs.h5"
)


re_evaluate_forgetting = False
if re_evaluate_forgetting:
    eval_interval = 1
    no_replay_forgetting_data = {}

    for run in noreplay_runs:
        rmses_over_epochs = run.evaluate_checkpoints_on_configs(
            generalization_test_config_dataset, evaluation_interval=eval_interval
        )
        no_replay_forgetting_data[run.N_samples] = rmses_over_epochs

    with open(f"{TRAINING_DIR}/no_replay_forgetting_data.yaml", "w") as f:
        yaml.dump(no_replay_forgetting_data, f)

    replay_forgetting_data = {}

    for run in replay_runs:
        replay_rmses_over_epochs = run.evaluate_checkpoints_on_configs(
            generalization_test_config_dataset, evaluation_interval=eval_interval
        )
        replay_forgetting_data[run.N_samples] = replay_rmses_over_epochs

    with open(f"{TRAINING_DIR}/replay_forgetting_data.yaml", "w") as f:
        yaml.dump(replay_forgetting_data, f)

    no_weight_decay_forgetting_data = {}

    for run in weight_decay_runs:
        no_weight_decay_rmses_over_epochs = run.evaluate_checkpoints_on_configs(
            generalization_test_config_dataset, evaluation_interval=1
        )
        if run.N_samples not in no_weight_decay_forgetting_data:
            no_weight_decay_forgetting_data[run.N_samples] = []
        no_weight_decay_forgetting_data[run.N_samples].append(
            {run.weight_decay: no_weight_decay_rmses_over_epochs}
        )

    with open(
        f"{TRAINING_DIR}/no_replay_no_weight_decay_forgetting_data.yaml", "w"
    ) as f:
        yaml.dump(no_weight_decay_forgetting_data, f)


else:
    with open(f"{TRAINING_DIR}/no_replay_forgetting_data.yaml") as f:
        no_replay_forgetting_data = yaml.safe_load(f)

    with open(f"{TRAINING_DIR}/replay_forgetting_data.yaml") as f:
        replay_forgetting_data = yaml.safe_load(f)

    with open(f"{TRAINING_DIR}/no_replay_no_weight_decay_forgetting_data.yaml") as f:
        no_weight_decay_forgetting_data = yaml.safe_load(f)


epoch_0_forgetting_loss = ModelTrainingRun.evaluate_model_on_config_set(
    mace_mp_calc, generalization_test_config_dataset
)


mace_omat_ood_test_loss = ModelTrainingRun.evaluate_model_on_config_set(
    omat_calc, generalization_test_config_dataset
)


print(f"Epoch Zero forgeting loss {epoch_0_forgetting_loss}")
print(f"Omat forgetting loss {mace_omat_ood_test_loss}")
forget_fig = plot_forgetting_loss(
    no_replay_forgetting_data=no_replay_forgetting_data,
    replay_forgetting_data=replay_forgetting_data,
    initial_test_set_loss=epoch_0_forgetting_loss,
)

plt.savefig("ForgettingFinetuning.pdf", dpi=300, bbox_inches="tight")


wd_forget_fig = plot_forgetting_loss_weight_decay(
    no_replay_forgetting_data=no_replay_forgetting_data,
    initial_test_set_loss=epoch_0_forgetting_loss,
    no_weight_decay_forgetting_data=no_weight_decay_forgetting_data,
)

plt.savefig("Weight_decayForgettingFinetuning.pdf", dpi=300, bbox_inches="tight")


# ===========================================================================
# Force evaluation on test set
phonon_test_mpids_file = (
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/test_spinel_mpids"
)

test_mpids_mat_iterator = FileMaterialsIterator(phonon_test_mpids_file)

test_pd_dataset = PhononDataset.load_phonon_dataset(
    test_mpids_mat_iterator, N_materials=3
)

pc = ConfigFactory(test_pd_dataset.phonon_data_samples)
configs = pc.construct_all_single_configs()
config_test_dataset = ConfigSingleDataset(configs)

no_replay_force_rmse = {}
replay_force_rmse = {}

for run in noreplay_runs:
    calc = run.get_mace_calculator()
    no_replay_force_rmse[run.N_samples] = run.evaluate_model_on_config_set(
        calc, config_test_dataset
    )


for run in replay_runs:
    calc = run.get_mace_calculator()
    replay_force_rmse[run.N_samples] = run.evaluate_model_on_config_set(
        calc, config_test_dataset
    )


mace_mp_force_rmse_test = run.evaluate_model_on_config_set(
    mace_mp_calc, config_test_dataset
)
omat_force_rmse = run.evaluate_model_on_config_set(omat_calc, config_test_dataset)

# Test set performance
print(mace_mp_force_rmse_test)
print(omat_force_rmse)
print(no_replay_force_rmse)
print(replay_force_rmse)


# ============================================================================
# Phonon FT Model Evaluation - Withheld Test set

phonon_test_mpids_file = (
    "/data/fast-pc-06/snw30/projects/phonons/phonotune/data/spinels/test_spinel_mpids"
)

test_mpids = list(FileMaterialsIterator(phonon_test_mpids_file))

td_maes_replay = []
phonon_mse_replay = []

td_maes_noreplay = []
phonon_mse_noreplay = []


reevaluate_phonons = False

replay_phonon_data_dicts = {}
noreplay_phonon_data_dicts = {}

if reevaluate_phonons:
    for tr in replay_runs:
        td_mae, phonon_mse = tr.evaluate_model_on_phonons(test_mpids)
        replay_phonon_data_dicts[tr.N_samples] = tr.phonon_evaluation.error_data

    for tr in noreplay_runs:
        td_mae, phonon_mse = tr.evaluate_model_on_phonons(test_mpids)
        noreplay_phonon_data_dicts[tr.N_samples] = tr.phonon_evaluation.error_data

else:
    for tr in replay_runs:
        with open(f"{tr.directory}/test_mp_ids_phonon_comp_{tr.model_name}.yaml") as f:
            data_dict = yaml.safe_load(f)
            replay_phonon_data_dicts[tr.N_samples] = data_dict

    for tr in noreplay_runs:
        with open(f"{tr.directory}/test_mp_ids_phonon_comp_{tr.model_name}.yaml") as f:
            data_dict = yaml.safe_load(f)
            noreplay_phonon_data_dicts[tr.N_samples] = data_dict


# ============================================================================
# Phonon TD evaluation on the withheld test set
mace_mp_med_benchmark = PhononBenchmark.construct_from_mpids(mace_mp_calc, test_mpids)

td_mae_dict = mace_mp_med_benchmark.calculate_thermodynamic_MAE()
phonon_mse, freq, errors = mace_mp_med_benchmark.compare_datasets_phonons()
print(f"Phonon MSE {phonon_mse}")
filepath = f"{TRAINING_DIR}/test_mp_ids_phonon_comp_mace_mp_medium.yaml"
mace_mp_med_benchmark.store_data(filepath, calculator_name="mace_mp_medium")

td_fig = plot_thermodynamic_property_errors(
    noreplay_phonon_data_dicts,
    replay_phonon_data_dicts,
    mace_mp_med_benchmark.error_data,
)

plt.savefig("td_prop_loss.pdf", dpi=300)


# ============================================================================
# Phonon Spectra of Spinell
ft_run_replay = replay_runs[-1]
print(ft_run_replay.model_name)


ft_run_noreplay = noreplay_runs[-1]
ft_weight_decay = next(
    filter(lambda x: x.N_samples == 2000 and x.weight_decay == 0.0, weight_decay_runs)
)

print(ft_weight_decay.model_name)
no_wd_ft_calc = MACECalculator(
    ft_weight_decay.model_file, device="cuda", enable_cueq=True
)

spinel_mpid = "mp-3536"

ft_calc_replay = MACECalculator(
    ft_run_replay.model_file, device="cuda", enable_cueq=True
)

ft_calc_no_replay = MACECalculator(
    ft_run_noreplay.model_file, device="cuda", enable_cueq=True
)


calculators = [omat_calc, mace_mp_calc, ft_calc_replay, ft_calc_no_replay]
calculator_names = ["OMAT", "MP-0 Med", "FT-replay", "FT-naive"]

reference_phonons = PhononData.create_phonopy_phonon_from_reference_alexandria_data(
    spinel_mpid
)
reference_phonons.auto_band_structure()


reference_bandstructure = reference_phonons.band_structure


fig, axes = plt.subplots(1, len(calculators), sharey=True, gridspec_kw={"wspace": 0})
fig.set_figwidth(6.5)
fig.set_figheight(3.5)
fig.set_tight_layout(True)


for ax, calc, calc_name in zip(axes, calculators, calculator_names, strict=False):
    unitcell = Unitcell.from_alexandria(spinel_mpid)
    unitcell = unitcell_fire_relaxation(unitcell, calc, relaxation_tolerance=0.005)

    phonons, _ = PhononData.create_phonopy_phonon_from_unitcell(unitcell, calc)
    phonons.auto_band_structure()

    if "naive" in calc_name:
        phonon_band_color = "tab:orange"
    elif "replay" in calc_name:
        phonon_band_color = "tab:blue"
    else:
        phonon_band_color = "tab:green"

    plot_model_reference_phonon_comparison(
        ax, phonons.band_structure, reference_bandstructure, phonon_band_color
    )

    ax.set_title(calc_name)


ylims_lo = []
ylims_hi = []
for ax in axes:
    ylim = ax.get_ylim()
    ylims_lo.append(ylim[0])
    ylims_hi.append(ylim[1])

new_lim_low = min(ylims_lo)
new_lim_hi = max(ylims_hi)

for ax in axes:
    ax.set_ylim(bottom=new_lim_low, top=new_lim_hi)


fig.supylabel("Phonon Frequency in THz")
fig.savefig("PhononCalc_comp.pdf", bbox_inches="tight")


fig, axes = plt.subplots(1, len(calculators), sharey=True, gridspec_kw={"wspace": 0})
fig.set_figwidth(6.5)
fig.set_figheight(3.5)

fig.set_tight_layout(True)
unitcell = Unitcell.from_alexandria(spinel_mpid)

for ax, calc, calc_name in zip(axes, calculators, calculator_names, strict=False):
    phonons, _ = PhononData.create_phonopy_phonon_from_unitcell(unitcell, calc)
    phonons.auto_band_structure()

    if "naive" in calc_name:
        phonon_band_color = "tab:orange"
    elif "replay" in calc_name:
        phonon_band_color = "tab:blue"
    else:
        phonon_band_color = "tab:green"

    plot_model_reference_phonon_comparison(
        ax, phonons.band_structure, reference_bandstructure, phonon_band_color
    )

    ax.set_title(calc_name)


ylims_lo = []
ylims_hi = []
for ax in axes:
    ylim = ax.get_ylim()
    ylims_lo.append(ylim[0])
    ylims_hi.append(ylim[1])

new_lim_low = min(ylims_lo)
new_lim_hi = max(ylims_hi)

for ax in axes:
    ax.set_ylim(bottom=new_lim_low, top=new_lim_hi)


fig.supylabel("Phonon Frequency in THz")
fig.savefig("Unrelaxed_PhononCalc_comp.pdf", bbox_inches="tight")
