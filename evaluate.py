from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig

from lgssl.datasets.builder import get_linearprobe_loaders
from lgssl.evaluation.fewshot import evaluate_fewshot
from lgssl.evaluation.linear_probe import evaluate_linear_probe
from lgssl.evaluation.utils import extract_features, get_model


@hydra.main(config_name="evaluation", config_path="./configs")
def main(cfg: DictConfig):
    model = get_model(cfg.model.name, cfg.model.checkpoint)

    assert any([cfg.evaluations[x] for x in cfg.evaluations]), "No evaluation tasks."
    lin_out = {}
    few_out = {}

    datasets = [
        "food101",
        "cifar10",
        "cifar100",
        "cub2011",
        "sun397",
        "cars196",
        "aircraft",
        "dtd",
        "pets",
        "caltech101",
        "flowers",
        "stl10",
        "eurosat",
        "resisc45",
        "pcam",
    ]
    if cfg.dataset.name == "all":
        eval_sets = datasets
    else:
        eval_sets = [cfg.dataset.name]

    logger.info(f"Evaluating {cfg.model.name} - {cfg.model.checkpoint}")
    num_sets = len(eval_sets)
    for i, dataset in enumerate(eval_sets):
        logger.info(f"Evaluating {dataset} -- {i}/{num_sets}")
        if dataset in ["aircraft", "pets", "caltech101", "flowers"]:
            use_mean_acc = True
        else:
            use_mean_acc = False

        train_loader, valid_loader, test_loader = get_linearprobe_loaders(
            dataset, cfg.dataset.image_mean
        )

        # extract features
        train_feats, train_labels = extract_features(model, train_loader)
        valid_feats, valid_labels = extract_features(model, valid_loader)
        test_feats, test_labels = extract_features(model, test_loader)
        num_classes = len(test_labels.unique())

        num_train = len(train_loader.dataset)
        num_valid = len(valid_loader.dataset)
        num_test = len(test_loader.dataset)

        logger.info(f"{dataset} | train: {num_train} val: {num_valid} test: {num_test}")
        logger.info(f"{dataset} | num classes: {num_classes}")

        if cfg.evaluations.linear_probe:
            lin_out[dataset] = evaluate_linear_probe(
                train_feats,
                train_labels,
                valid_feats,
                valid_labels,
                test_feats,
                test_labels,
                use_mean_acc,
                max_iter=cfg.logistic_regression.max_iter,
                combine_trainval=cfg.logistic_regression.combine_trainval,
                use_sklearn=cfg.logistic_regression.use_sklearn,
            )
            logger.info(f"Evaluated {dataset} ({i}/{num_sets}): {lin_out[dataset]:.2f}")

        if cfg.evaluations.fewshot and dataset != "pcam":
            few_out[dataset] = evaluate_fewshot(
                train_feats, train_labels, test_feats, test_labels, cfg.fewshot
            )
        else:
            few_out[dataset] = (-1, -1)

    logger.info(f"Model: {cfg.model.name} || {cfg.model.checkpoint}")
    logger.info("Datasets:        " + " & ".join([f"{x:>10}" for x in eval_sets]))
    if cfg.evaluations.linear_probe:
        lin_mean = " & ".join([f"{lin_out[x]:10.1f}" for x in eval_sets])
        logger.info(f"linear probe:    {lin_mean}")

    if cfg.evaluations.fewshot:
        few_mean = " & ".join([f"{few_out[x][0]:10.1f}" for x in eval_sets])
        few_std = " & ".join([f"{few_out[x][1]:10.1f}" for x in eval_sets])
        logger.info(f"Fewshot mean:    {few_mean}")
        logger.info(f"Fewshot std:     {few_std}")


if __name__ == "__main__":
    main()
