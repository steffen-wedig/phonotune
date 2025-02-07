import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from phonotune.calculation_configuration import Material
from phonotune.plotting_bands import extract_bands, load_yaml_data, plot_phonon_bands

# Define a color for each calculator.
calc_colors = {"MACE_OMAT": "black", "MACE_MP_0": "red"}

# Define materials.
mat1 = Material(name="Mn4Si7", temperature="high")
mat2 = Material(name="Mn4Si7", temperature="low")
mat3 = Material(name="Ru2Sn3", temperature="high")
mat4 = Material(name="Ru2Sn3", temperature="low")
mats = [mat1, mat2, mat3, mat4]


N_mats = len(mats)
N_bands = 3

# Create a figure with 4 subplots (one per material).
fig, axes = plt.subplots(1, 4, figsize=(5 * N_mats, 6))
# Loop over the materials (each gets its own axis)
for idx, mat in enumerate(mats):
    ax = axes[idx]

    ax.set_title(f"{mat.name} ({mat.temperature})")
    ax.set_xlabel("x")
    ax.set_ylabel("f in THz")
    # Loop over calculators, plotting each set of bands in its assigned color.
    for calc, color in calc_colors.items():
        runname = f"{mat.name}_{mat.temperature}_{calc}"
        try:
            data = load_yaml_data(f"phonons/{runname}_bands.yaml")
        except FileNotFoundError:
            print(f"No phonons for {runname}")
            continue
        bands = extract_bands(data, N_bands)
        plot_phonon_bands(bands, ax=ax, color=color)

# Create custom legend handles for the calculators.
legend_handles = [
    Line2D([0], [0], color=color, lw=2, label=calc)
    for calc, color in calc_colors.items()
]
# Add a global legend to the figure.
fig.legend(handles=legend_handles, loc="upper right")


plt.savefig("fine_relaxed.png")
