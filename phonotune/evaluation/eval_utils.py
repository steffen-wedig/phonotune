import json
import re


def parse_training_results(path: str) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            results.append(d)

    return results


def get_model_name_epoch_from_checkpoint_path(filepath):
    pattern = r"^(.+_\d+)_(?:wrp_)?run-\d+_epoch-(\d+)\.pt$"

    match = re.search(pattern, filepath)
    if match:
        model_name = match.group(1)
        epoch = int(match.group(2))
        return model_name, epoch
    else:
        raise ValueError("The filepath does not match the expected format.")


def get_all_checkpoint_files_in_run_dir(run_directory: str):
    raise NotImplementedError


def extract_weight_decay(model_name: str) -> float:
    pattern = r"weight_decay_((\d+)e_(\d+)|(\d+))"
    match = re.search(pattern, model_name)
    if not match:
        raise ValueError(
            "The model name does not contain a valid weight_decay pattern."
        )

    # If we matched the pattern with exponent
    if match.group(2) and match.group(3):
        coeff = float(match.group(2))  # adjust coefficient (e.g. 25 -> 2.5)
        if coeff > 10:
            coeff = coeff / 10
        exponent = int(match.group(3))
        return coeff * (10 ** (-exponent))
    elif match.group(4):  # plain number
        value = match.group(4)
        # If the number is "0", return 0. Otherwise apply the same division logic.
        return 0.0 if value == "0" else float(value) / 10.0
