from collections.abc import Sequence

import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np

from phonotune.evaluation.training_evaluation import ModelTrainingRun

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for all text rendering
        "font.family": "serif",  # Use a serif font like Computer Modern
        "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
        "axes.labelsize": 12,  # Axis label font size
        "axes.titlesize": 14,  # Title font size
        "legend.fontsize": 10,  # Legend font size
        "xtick.labelsize": 10,  # X-axis tick labels
        "ytick.labelsize": 10,  # Y-axis tick labels
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",  # Extra LaTeX packages
    }
)


def plot_validation_loss_curves_over_epoch(
    noreplay_runs: Sequence[ModelTrainingRun], replay_runs: Sequence[ModelTrainingRun]
):
    replay_orig_cmap = plt.cm.Blues
    cNorm = matplotlib.colors.LogNorm(vmin=5, vmax=2000)
    replay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=replay_orig_cmap)

    noreplay_orig_cmap = plt.cm.Oranges
    noreplay_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=noreplay_orig_cmap)

    fig, ax = plt.subplots(figsize=(8, 6))

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
    ax.set_xlim(left=0, right=max(epochs))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch Number", fontsize=14)
    ax.set_ylabel("Validation Loss Force RMSE in meV/A", fontsize=14)
    ax.set_title("Validation Loss Over Epochs", fontsize=16)
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.set_xticks(np.linspace(0, 20, 5, dtype=int))
    # Improve layout
    plt.tight_layout()

    return fig


def plot_val_loss_over_N_samples(replay_min_val_loss, noreplay_min_val_loss):
    fig = plt.figure(figsize=(8, 6))

    # Convert dataset size and loss values to arrays for better handling
    x_replay = np.array(list(replay_min_val_loss.keys()))
    y_replay = np.array(list(replay_min_val_loss.values()))

    x_noreplay = np.array(list(noreplay_min_val_loss.keys()))
    y_noreplay = np.array(list(noreplay_min_val_loss.values()))

    # Use distinct markers for better visibility
    plt.loglog(
        x_replay, y_replay, marker="o", linestyle="-", label="Replay", color="tab:blue"
    )
    plt.loglog(
        x_noreplay,
        y_noreplay,
        marker="o",
        linestyle="-",
        label="No Replay",
        color="tab:orange",
    )

    # Labels & Title
    plt.xlabel("Dataset Size", fontsize=14)
    plt.ylabel("Validation Loss (log scale)", fontsize=14)
    plt.title("Validation Loss vs Dataset Size", fontsize=16)

    # Legend & Layout
    plt.legend(fontsize=10, frameon=False)
    plt.tight_layout()

    return fig
