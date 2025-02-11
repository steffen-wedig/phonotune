import matplotlib.pyplot as plt
import yaml


def load_yaml_data(filename):
    """Load phonon data from a YAML file."""
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


def extract_bands(data, N_bands):
    """
    Extract bands from the phonon data.

    The YAML data is assumed to have a top-level key 'phonon' with each element
    corresponding to a q-point that contains a 'distance' and a list of 'band' dictionaries.

    Returns:
        bands: list of lists, where each inner list corresponds to one band and is
               a list of (distance, frequency) tuples.
    """
    phonon_points = data.get("phonon", [])
    if not phonon_points:
        raise ValueError("No phonon data found in the YAML file!")

    # Determine the number of bands from the first q-point.
    num_bands = len(phonon_points[0].get("band", []))

    assert N_bands < num_bands
    # Create a list for each band.
    bands = [[] for _ in range(N_bands)]

    # Loop over each q-point.
    for point in phonon_points:
        d = point.get("distance", None)
        if d is None:
            raise ValueError("Missing 'distance' in one of the phonon points.")

        band_list = point.get("band", [])
        if len(band_list) != num_bands:
            raise ValueError("Inconsistent number of bands across q-points.")

        # Append a tuple (distance, frequency) to the corresponding band list.
        for i, band in enumerate(band_list[:N_bands]):
            freq = band.get("frequency", None)
            if freq is None:
                raise ValueError(f"Missing 'frequency' in band {i + 1} at distance {d}")
            bands[i].append((d, freq))

    return bands


def plot_phonon_bands(bands, ax: plt.Axes = None, color: str = "k"):
    """
    Plot the phonon bands.

    Parameters:
      bands : list of bands, where each band is a list of (distance, frequency) tuples.
    """

    if ax is None:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 6)

    # Plot each band as a line.
    for band in bands:
        # Unzip the list of tuples into separate lists for distances and frequencies.
        distances, frequencies = zip(*band, strict=False)
        ax.plot(distances, frequencies, c=color)
    return ax


if __name__ == "__main__":
    data = load_yaml_data(
        "/data/fast-pc-06/snw30/projects/phonons/phonotune/phonons/Mn4Si7_high_MACE_OMAT_bands.yaml"
    )
    bands = extract_bands(data, 5)
    fig, ax = plt.subplots(1)
    plot_phonon_bands(bands, ax)
    fig.savefig("testfig")
