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
    pass
