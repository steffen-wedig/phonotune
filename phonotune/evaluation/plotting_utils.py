from collections.abc import Sequence

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from phonotune.evaluation.training_evaluation import ModelTrainingRun

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for all text rendering
        "font.family": "serif",  # Use a serif font like Computer Modern
        "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
        "axes.labelsize": 11,  # Axis label font size
        "axes.titlesize": 14,  # Title font size
        "legend.fontsize": 7,  # Legend font size
        "xtick.labelsize": 10,  # X-axis tick labels
        "ytick.labelsize": 10,  # Y-axis tick labels
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",  # Extra LaTeX packages
    }
)


def plot_validation_loss_curves_over_epoch(
    noreplay_runs: Sequence[ModelTrainingRun], replay_runs: Sequence[ModelTrainingRun]
) -> Figure:
    replay_orig_cmap = plt.cm.Blues
    cNorm = matplotlib.colors.LogNorm(vmin=5, vmax=2000)
    replay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=replay_orig_cmap)

    noreplay_orig_cmap = plt.cm.Oranges
    noreplay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=noreplay_orig_cmap)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot fine-tuning without replay
    for _, tr in enumerate(noreplay_runs):
        epochs = np.arange(0, len(tr.validation_loss))
        val_loss = np.array([tr.validation_loss[ep] for ep in epochs])
        colorVal = noreplay_scalarMap.to_rgba(tr.N_samples)
        ax.plot(
            epochs,
            val_loss,
            "-x",
            color=colorVal,
            label=f"No replay, N = {tr.N_samples}",
        )

    # Reset color cycle for second set of plots
    ax.set_prop_cycle(None)

    # Plot fine-tuning with replay
    for _, tr in enumerate(replay_runs):
        epochs = np.arange(0, len(tr.validation_loss))
        val_loss = np.array([tr.validation_loss[ep] for ep in epochs])
        colorVal = replay_scalarMap.to_rgba(tr.N_samples)

        ax.plot(
            epochs, val_loss, "-x", color=colorVal, label=f"Replay, N = {tr.N_samples}"
        )

    # Formatting
    max_epochs = max(epochs)
    ax.set_xlim(left=0, right=max_epochs)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Validation Set Force RMSE ($\mathrm{meV/\AA}$)")
    # ax.set_title("Validation Loss Over Epochs", fontsize=16)
    ax.legend(loc="best", frameon=False, ncol=2)
    ax.set_xticks(np.linspace(0, max_epochs, int(max_epochs / 5) + 1, dtype=int))
    # Improve layout
    plt.tight_layout()

    return fig


def plot_forgetting_loss(
    no_replay_forgetting_data, replay_forgetting_data, initial_test_set_loss
) -> Figure:
    replay_orig_cmap = plt.cm.Blues
    cNorm = matplotlib.colors.LogNorm(vmin=5, vmax=2000)
    replay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=replay_orig_cmap)

    noreplay_orig_cmap = plt.cm.Oranges
    noreplay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=noreplay_orig_cmap)

    fig, ax = plt.subplots(figsize=(6, 4))

    print(no_replay_forgetting_data.items())

    # Plot fine-tuning without replay
    for N_samples, loss_dict in no_replay_forgetting_data.items():
        print(loss_dict.keys())
        epochs = [epoch for epoch in loss_dict.keys()]
        forget_loss = [loss_val for loss_val in loss_dict.values()]

        epochs.append(0)
        forget_loss.append(initial_test_set_loss)

        epochs = np.array(epochs)
        forget_loss = np.array(forget_loss)

        sort_idx = np.argsort(epochs)
        epochs = epochs[sort_idx]
        forget_loss = forget_loss[sort_idx]
        print(N_samples)
        print(epochs)
        print(forget_loss)
        colorVal = noreplay_scalarMap.to_rgba(N_samples)
        ax.plot(
            epochs,
            forget_loss,
            "-x",
            color=colorVal,
            label=f"No replay, N = {N_samples}",
        )

    for N_samples, loss_dict in replay_forgetting_data.items():
        epochs = [epoch for epoch in loss_dict.keys()]
        forget_loss = [loss_val for loss_val in loss_dict.values()]
        epochs.append(0)
        forget_loss.append(initial_test_set_loss)

        epochs = np.array(epochs)
        forget_loss = np.array(forget_loss)

        sort_idx = np.argsort(epochs)
        epochs = epochs[sort_idx]
        forget_loss = forget_loss[sort_idx]

        colorVal = replay_scalarMap.to_rgba(N_samples)

        ax.plot(
            epochs, forget_loss, "-x", color=colorVal, label=f"Replay, N = {N_samples}"
        )

    max_epochs = max(epochs)
    ax.set_xlim(left=0, right=max_epochs)
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel(r"Forgetting Test Loss Force RMSE ($\mathrm{meV/\AA}$)")
    # ax.set_title("Finetuning Forgetting Loss Over Epochs", fontsize=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis

    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 0.5))
    ax.set_xticks(np.linspace(0, max_epochs, int(max_epochs / 5) + 1, dtype=int))

    return fig


def plot_val_loss_over_N_samples(noreplay_min_val_loss, replay_min_val_loss) -> Figure:
    fig = plt.figure(figsize=(2.75, 4))

    # Convert dataset size and loss values to arrays for better handling
    x_replay = np.array(list(replay_min_val_loss.keys()))
    y_replay = np.array(list(replay_min_val_loss.values()))

    x_noreplay = np.array(list(noreplay_min_val_loss.keys()))
    y_noreplay = np.array(list(noreplay_min_val_loss.values()))

    # Use distinct markers for better visibility
    plt.loglog(
        x_replay, y_replay, marker="x", linestyle="-", label="Replay", color="tab:blue"
    )
    plt.loglog(
        x_noreplay,
        y_noreplay,
        marker="x",
        linestyle="-",
        label="No Replay",
        color="tab:orange",
    )

    # Labels & Title
    plt.xlabel("Dataset Size")
    plt.ylabel(r"Val. Min. Force RMSE ($\mathrm{meV/\AA}$)")
    # plt.title("Validation Loss vs Dataset Size", fontsize=16)

    # Legend & Layout
    plt.legend(frameon=False)
    plt.tight_layout()

    return fig


def plot_thermodynamic_property_errors(
    noreplay_phonon_data_dicts, replay_phonon_data_dicts, baseline_data
):
    td_keys = baseline_data["td_maes"].keys()

    N_properties = len(td_keys)
    fig, axes = plt.subplots(
        N_properties,
        1,
        figsize=(
            4,
            5 * N_properties,
        ),
    )

    for ax, td_property in zip(axes, td_keys, strict=False):
        N_samples_noreplay = []
        errors_noreplay = []

        for N_samples, noreplay_run_data in noreplay_phonon_data_dicts.items():
            N_samples_noreplay.append(N_samples)
            errors_noreplay.append(noreplay_run_data["td_maes"][td_property])

        N_samples_noreplay = np.array(N_samples_noreplay)
        errors_noreplay = np.array(errors_noreplay)

        sort_idx = np.argsort(N_samples_noreplay)
        N_samples_noreplay = N_samples_noreplay[sort_idx]
        errors_noreplay = errors_noreplay[sort_idx]

        ax.plot(
            N_samples_noreplay,
            errors_noreplay,
            marker="x",
            linestyle="-",
            label="Replay",
            color="tab:orange",
        )

        N_samples_replay = []
        errors_replay = []

        for N_samples, replay_run_data in replay_phonon_data_dicts.items():
            N_samples_replay.append(N_samples)
            errors_replay.append(replay_run_data["td_maes"][td_property])

        N_samples_replay = np.array(N_samples_replay)
        errors_replay = np.array(errors_replay)

        sort_idx = np.argsort(N_samples_replay)
        N_samples_replay = N_samples_replay[sort_idx]
        errors_replay = errors_replay[sort_idx]

        ax.plot(
            N_samples_replay,
            errors_replay,
            marker="x",
            linestyle="-",
            label="Replay",
            color="tab:blue",
        )

        ax.set_title(td_property)

        ax.hlines(
            baseline_data["td_maes"][td_property],
            xmin=0,
            xmax=2000,
            color="k",
            linestyle="--",
        )

    return fig
