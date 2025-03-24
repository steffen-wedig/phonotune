import torch
from mace.calculators import MACECalculator


# Taken from Lars' MACE PR https://github.com/ACEsuit/mace/pull/845
def update_weights_from_checkpoint(mace_calculator: MACECalculator, checkpoint_path):
    """
    Updates the model's weights from a given checkpoint file.
    Args:
        checkpoint_path (str): Path to the checkpoint file.
    Example:
        >>> atoms = read('path/to/atoms_file.xyz')
        >>> atoms.calc = MACECalculator('path/to/mace_weights', device=device)
        >>> print("Energy before:", atoms.get_potential_energy())
        >>> atoms.calc.update_weights_from_checkpoint('path/to/checkpoint.pt')
        >>> print("Energy after:", atoms.get_potential_energy())
    """

    assert len(mace_calculator.models) == 1, (
        "Checkpoint update only supported for single model, "
        f"not committee of {len(mace_calculator.models)} models"
    )

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=mace_calculator.device)[
        "model"
    ]

    state_dict = checkpoint.get("state_dict", checkpoint)

    # Load the weights into the model
    mace_calculator.models[0].load_state_dict(state_dict, strict=True)
    mace_calculator.models[0].eval()  # Ensure the model stays in evaluation mode
    # mace_calculator.reset()  # Clear ase calculator cache
    return mace_calculator
