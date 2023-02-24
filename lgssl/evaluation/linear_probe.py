# Based on evaluate_zeroshot from SLIP but changed by MB
from __future__ import annotations

import random
import time
from collections import defaultdict
from math import log10
from warnings import simplefilter

import torch
import torch.utils.data
from loguru import logger
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression

from lgssl.evaluation.logistic_regression import LogisticRegression
from lgssl.utils.metrics import Accuracy

# Silence repeated convergence warnings from scikit-learn logistic regression.
simplefilter("ignore", category=ConvergenceWarning)


def evaluate_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    test_feats,
    test_labels,
    use_mean_accuracy,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    start = time.time()
    classifier = train_linear_probe(
        train_feats,
        train_labels,
        valid_feats,
        valid_labels,
        use_mean_accuracy,
        sk_verbose,
        max_iter=max_iter,
        combine_trainval=combine_trainval,
        use_sklearn=use_sklearn,
    )
    test_acc = test_linear_probe(classifier, test_feats, test_labels, use_mean_accuracy)

    del classifier
    torch.cuda.empty_cache()
    print(f"Time taken {time.time() - start:.2f}")
    return test_acc


def train_linear_probe(
    train_feats,
    train_labels,
    valid_feats,
    valid_labels,
    use_mean_accuracy,
    sk_verbose=False,
    max_iter=100,
    combine_trainval=True,
    use_sklearn=False,
):
    """
    Args:
        holdout_fraction: Fraction of the (official) train split to hold out for
            validation while searching the cost hyperparameter. Holding out will
            be deterministic and have similar class distribution as train split.
    """
    NUM_C = len(set(train_labels.cpu().numpy()))
    acc_meter = Accuracy(num_classes=NUM_C, mean_per_class=use_mean_accuracy)

    if valid_labels is None:
        trainval_feats = train_feats
        trainval_labels = train_labels
        train_ind, valid_ind = split_trainval(trainval_labels.cpu().numpy(), 0.2)
        train_feats = trainval_feats[train_ind]
        train_labels = trainval_labels[train_ind]
        valid_feats = trainval_feats[valid_ind]
        valid_labels = trainval_labels[valid_ind]

    # CLIP performed linear probe evaluation by sweeping over 96 log-spaced costs.
    # Following CLIP, we sweep in two stages (coarse and fine) for quick search.
    costs = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
    logger.info(f"First sweep with costs: {costs}")

    # Train and avaluate each classifier and get accuracy.
    accuracies = []
    for cost in costs:
        classifier = _fit_logreg(
            train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn
        )
        predictions = classifier.predict_proba(valid_feats)
        accuracy = acc_meter(torch.as_tensor(predictions), valid_labels)
        accuracies.append(accuracy)

        acc_meter.reset()
        logger.info(f"Cost = {cost}, Top-1 accuracy = {accuracy:.3f}")

    # Second sweep: search around the best cost with a resolution of 8 steps per
    # decade. Example: if best cost = 1e2, then these costs will be in (1, 1e-4).
    best_accuracy = max(accuracies)
    best_cost = costs[accuracies.index(best_accuracy)]
    costs = torch.logspace(log10(best_cost) - 2, log10(best_cost) + 2, 29)
    costs = costs[(costs >= 1e-6) & (costs <= 1e6)].tolist()

    # We may visit the same cost value multiple times while searching, to avoid
    # re-training the classifier, keep a map of accuracies per cost.
    accuracies = {best_cost: best_accuracy}

    logger.info("Performing second sweep as a binary search around best cost.")
    logger.info(f"Initial search space: {[round(c, 3) for c in costs]}")

    while len(costs) > 1:
        # Get mid-points of left/right half interval of search space: (25,50,75)%
        cost_25 = costs[len(costs) // 4]
        cost_50 = costs[len(costs) // 2]
        cost_75 = costs[-len(costs) // 4]
        logger.info(
            f"Half interval mid-points: {cost_25=:.3f}, {cost_50=:.3f}, {cost_75=:.3f}"
        )

        # Compute accuracy for these costs (skip if computed in prev iteration).
        for cost in [cost_25, cost_50, cost_75]:
            _acc = accuracies.get(cost, None)
            if _acc is None:
                classifier = _fit_logreg(
                    train_feats, train_labels, cost, sk_verbose, max_iter, use_sklearn
                )
                predictions = classifier.predict_proba(valid_feats)
                _acc = acc_meter(torch.as_tensor(predictions), valid_labels)
                accuracies[cost] = _acc
                acc_meter.reset()

            logger.info(f"Cost = {round(cost, 3)}, Top-1 accuracy = {_acc:.3f}")

        # Cut down the search space by half such that the mid-point of the resulting
        # reduced search space is the cost with current best accuracy.
        max_acc = max(accuracies[cost_25], accuracies[cost_50], accuracies[cost_75])
        costs = (
            costs[: len(costs) // 2]
            if max_acc == accuracies[cost_25]
            else costs[len(costs) // 2 :]
            if max_acc == accuracies[cost_75]
            else costs[len(costs) // 4 : -len(costs) // 4]
        )
        logger.info(f"Reduced search space, costs: {[round(c, 3) for c in costs]}")

    # Filter None accuracy values (some costs may not be visited while searching).
    # Then find best accuracy and its cost.
    best_cost, best_accuracy = max(accuracies.items(), key=lambda k: k[1])
    logger.info(f"Best cost = {best_cost:.3f}, Top-1 accuracy = {best_accuracy:.3f}")

    # train final classifier
    if combine_trainval:
        trainval_feats = torch.cat([train_feats, valid_feats], dim=0)
        trainval_labels = torch.cat([train_labels, valid_labels], dim=0)

        final_classifier = _fit_logreg(
            trainval_feats,
            trainval_labels,
            best_cost,
            sk_verbose,
            max_iter,
            use_sklearn,
        )
    else:
        final_classifier = _fit_logreg(
            train_feats, train_labels, best_cost, sk_verbose, max_iter, use_sklearn
        )

    return final_classifier


def test_linear_probe(
    linear_classifier, test_feats, test_labels, use_mean_accuracy, num_classes=None
):
    # evaluate
    NUM_C = len(set(test_labels.cpu().numpy())) if num_classes is None else num_classes
    acc_meter = Accuracy(num_classes=NUM_C, mean_per_class=use_mean_accuracy)
    predictions = linear_classifier.predict_proba(test_feats)
    accuracy = acc_meter(torch.as_tensor(predictions), test_labels)

    logger.info(f"Test accuracy: {accuracy:.3f}")
    return accuracy


def _fit_logreg(
    feats: torch.Tensor,
    labels: torch.Tensor,
    cost: float,
    verbose: bool = False,
    max_iter: int = 100,
    use_sklearn: bool = False,
) -> LogisticRegression:
    """
    Initialize and fit a `LogisticRegression` classifier for input features and
    labels. Default settings follow CLIP (L-BFGS, 1K iterations, etc.).
    """
    if use_sklearn:
        classifier = sk_LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose, random_state=0
        )
    else:
        classifier = LogisticRegression(
            C=cost, max_iter=max_iter, verbose=verbose, random_state=0
        )
    classifier.fit(feats, labels)
    return classifier


def split_trainval(targets, val_percentage):
    # Organize dataset by classes (class ID -> list[dataset index] map).
    labels_to_indices = defaultdict(list)
    for index, label in enumerate(targets):
        labels_to_indices[label].append(index)

    train_indices = []
    valid_indices = []
    for label, indices in labels_to_indices.items():
        # Deterministic shuffling to ensure same held-out split across runs.
        random.Random(93).shuffle(indices)

        train_indices.extend(indices[int(len(indices) * val_percentage) :])
        valid_indices.extend(indices[: int(len(indices) * val_percentage)])

    return train_indices, valid_indices
