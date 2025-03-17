from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from phonopy.phonon.band_structure import BandStructure


@dataclass
class Phononband:
    distances: np.ndarray
    frequencies: np.ndarray


def get_bands_from_bandstructure(band_structure: BandStructure) -> Sequence[Phononband]:
    N_intervals = len(band_structure.distances)

    bands = []

    f = None
    d = None
    for i in range(N_intervals):
        if f is None:
            f = band_structure.frequencies[i].T
            d = band_structure.distances[i]
        else:
            f = np.hstack((f, band_structure.frequencies[i].T))
            d = np.hstack((d, band_structure.distances[i]))
        if band_structure.path_connections[i] is False:
            for i in range(len(f)):
                bands.append(Phononband(distances=d, frequencies=f[i]))
            f = None
            d = None

    return bands


def get_BZ_x_axis_ticks(band_structure: BandStructure):
    q_labels = band_structure.labels

    path_connections = band_structure.path_connections

    positions = [pos[0] for pos in band_structure.distances]
    positions.append(band_structure.distances[-1][-1])

    ticks = [positions[0] + 0.01]  # Always mark the start
    tick_labels = [q_labels[0]]

    pos_iter = iter(positions[1:])
    label_iter = iter(q_labels[1:])
    # Loop over the internal connections. The index i here corresponds to
    # the q point at the end of the i-th segment (and the start of the next)
    for i, connected in enumerate(path_connections):
        pos = next(pos_iter)
        print(pos)

        if i == len(path_connections) - 1:
            pos = pos - 0.01
        label = next(label_iter)

        if connected or i == len(path_connections) - 1:
            ticks.append(pos)
            tick_labels.append(label)
        else:
            ticks.append(pos)
            tick_labels.append(f"{label}{next(label_iter)}")

    # Always mark the final q point (i.e. the end of the last segment)

    return np.array(ticks), tick_labels


def plot_bands(bandstructure: BandStructure):
    fig, ax = plt.subplots()

    bands = get_bands_from_bandstructure(band_structure=bandstructure)

    for band in bands:
        ax.plot(band.distances, band.frequencies, color="k", linewidth=0.5)

    ticks, tick_labels = get_BZ_x_axis_ticks(band_structure=bandstructure)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(
        left=bandstructure.distances[0][0], right=bandstructure.distances[-1][-1]
    )

    plt.ylabel("Frequency (THz)")
    return fig


def plot_model_reference_phonon_comparison(
    ax: plt.Axes,
    model_bandstructure: BandStructure,
    ref_bandstructure: BandStructure,
    phonon_band_color,
):
    assert model_bandstructure.labels == ref_bandstructure.labels

    ref_bands = get_bands_from_bandstructure(band_structure=ref_bandstructure)
    model_bands = get_bands_from_bandstructure(band_structure=model_bandstructure)

    print(len(ref_bands))
    print(len(model_bands))

    for band in ref_bands:
        ax.plot(band.distances, band.frequencies, color="k", label="Reference")

    for band in model_bands:
        ax.plot(
            band.distances,
            band.frequencies,
            color=phonon_band_color,
            label="Prediction",
        )

    ticks, tick_labels = get_BZ_x_axis_ticks(band_structure=ref_bandstructure)
    #
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7)
    x_min = ref_bandstructure.distances[0][0]
    x_max = ref_bandstructure.distances[-1][-1]

    ax.hlines(y=0, xmin=x_min, xmax=x_max, colors="k", alpha=0.5)
    ax.set_xlim(left=x_min, right=x_max)
