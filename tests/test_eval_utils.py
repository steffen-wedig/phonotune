from phonotune.evaluation.eval_utils import extract_weight_decay


def test_extract_weight_decay():
    model_names = [
        "mace_single_force_finetune_config_weight_decay_25e_7-2000",
        "mace_single_force_finetune_config_weight_decay_1e_6-2000",
        "mace_single_force_finetune_config_weight_decay_0-2000",
    ]

    weight_decay_gt = [2.5e-7, 1e-6, 0.0]

    for name, gt in zip(model_names, weight_decay_gt, strict=False):
        weight_decay = extract_weight_decay(name)
        print(weight_decay)
        assert gt == weight_decay


test_extract_weight_decay()
