"""Fit comparable AF classifiers and calibrate them without held-out subjects."""

from collections.abc import Callable, Mapping
from math import isfinite
from operator import index

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _subject_folds(labels, subjects, n_splits, seed):
    """Spread AF-only, mixed, non-AF-only, and unlabeled groups over folds."""
    names, subject_rows = np.unique(subjects, return_inverse=True)
    if not 2 <= n_splits <= len(names):
        raise ValueError("Fold count must be between 2 and the number of participants")

    af = np.zeros(len(names), dtype=bool)
    non_af = np.zeros_like(af)
    np.logical_or.at(af, subject_rows, labels == 1)
    np.logical_or.at(non_af, subject_rows, labels == 0)
    rng = np.random.default_rng(seed)
    subject_folds = np.empty(len(names), dtype=int)
    position = 0
    for has_af, has_non_af in ((True, False), (True, True), (False, True), (False, False)):
        group = rng.permutation(np.flatnonzero((af == has_af) & (non_af == has_non_af)))
        subject_folds[group] = np.arange(position, position + len(group)) % n_splits
        position += len(group)
    return subject_folds[subject_rows]


def _both_classes(label_values):
    return bool(np.any(label_values == 0) and np.any(label_values == 1))


def _fit_and_predict(features, labels, train_rows, score_rows, *, regularization):
    predictions = np.full(len(score_rows), np.nan)
    available = np.isfinite(features[score_rows]).all(axis=1)
    if len(train_rows) and np.any(available):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, class_weight="balanced", max_iter=1000),
        )
        classifier.fit(features[train_rows], labels[train_rows])
        predictions[available] = classifier.predict_proba(features[score_rows[available]])[:, 1]
    return predictions


def _nested_folds(train_rows, fit_labels, subjects, masks, max_splits, seed):
    """Find shared inner folds with the greatest route support for calibration."""
    best = None
    for n_splits in range(min(max_splits, len(np.unique(subjects[train_rows]))), 1, -1):
        assignment = _subject_folds(fit_labels[train_rows], subjects[train_rows], n_splits, seed)
        supported = {
            route for route, mask in masks.items()
            if all(_both_classes(fit_labels[train_rows[(assignment != fold) & mask[train_rows]]])
                   for fold in range(n_splits))
        }
        if best is None or len(supported) > len(best[2]):
            best = assignment, n_splits, supported
        if len(supported) == len(masks):
            break
    return best if best is not None else (None, 0, set())


def train_oof(
    features: Mapping[str, np.ndarray], labels, subjects, *, n_splits=5, inner_splits=3,
    seed=2026, train_eligible=None, calibration_selector: Callable | None = None, regularization=1.0,
) -> dict:
    """Fit one logistic model per route on the same outer participant folds.

    Labels are 0 (non-AF), 1 (AF), or -1 (excluded from classifier fitting).
    A route abstains when any feature in the row is nonfinite. If provided,
    ``train_eligible`` further limits which labeled rows can fit a model.
    Eligible and excluded rows can both receive held-out scores when available.

    ``calibration_selector(scores, rows)`` receives predictions and original
    row indices for the outer training participants. Every calibration score
    was produced on an inner held-out participant. Return a dictionary with
    ``threshold`` and any calibration diagnostics. A missing threshold produces
    unavailable decisions, with the raw held-out scores still returned.
    ``regularization`` is the positive logistic regression C in every fit.
    """
    if not isfinite(regularization) or regularization <= 0:
        raise ValueError("Logistic regression C must be positive and finite")
    labels = np.asarray(labels)
    subjects = np.asarray(subjects, dtype=str)
    if (labels.ndim != 1 or not len(labels) or not np.isin(labels, (-1, 0, 1)).all()
            or subjects.shape != labels.shape or np.any(subjects == "")):
        raise ValueError("Expected aligned participant IDs and labels in {-1, 0, 1}")
    labels = labels.astype(int)
    if train_eligible is not None:
        train_eligible = np.asarray(train_eligible)
        if train_eligible.shape != labels.shape or train_eligible.dtype != bool:
            raise ValueError("train_eligible must contain one boolean per row")
    fit_labels = labels if train_eligible is None else np.where(train_eligible, labels, -1)
    if any(len(np.unique(subjects[fit_labels == label])) < 2 for label in (0, 1)):
        raise ValueError("Grouped training needs each AF class in at least two participants")

    matrices = {route: np.asarray(values, dtype=float) for route, values in features.items()}
    if not matrices or any(x.ndim != 2 or x.shape[0] != len(labels) or not x.shape[1] for x in matrices.values()):
        raise ValueError("Each route needs a feature matrix aligned to the labeled rows")
    masks = {route: (fit_labels >= 0) & np.isfinite(x).all(axis=1) for route, x in matrices.items()}
    n_splits, inner_splits = index(n_splits), index(inner_splits)
    if inner_splits < 2:
        raise ValueError("At least two inner folds are needed for grouped calibration")
    folds = _subject_folds(fit_labels, subjects, n_splits, seed)
    routes = {route: {"scores": np.full(len(labels), np.nan),
                      "decisions": np.full(len(labels), -1, dtype=np.int8), "folds": []}
              for route in matrices}

    for fold in range(n_splits):
        training = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        inner, inner_count, supported = (None, 0, set())
        if calibration_selector is not None:
            inner, inner_count, supported = _nested_folds(
                training, fit_labels, subjects, masks, inner_splits, seed + fold + 1,
            )
        for route, matrix in matrices.items():
            history = routes[route]
            fit = training[masks[route][training]]
            option = {"threshold": None, "status": "not_calibrated"}
            if not _both_classes(labels[fit]):
                option["status"] = "insufficient_training_labels"
            else:
                history["scores"][test] = _fit_and_predict(matrix, labels, fit, test,
                                                           regularization=regularization)
                if calibration_selector is not None:
                    option["status"] = "insufficient_calibration_training_labels"
                if route in supported:
                    calibration = np.full(len(training), np.nan)
                    for inner_fold in range(inner_count):
                        inner_fit = training[(inner != inner_fold) & masks[route][training]]
                        inner_test = inner == inner_fold
                        calibration[inner_test] = _fit_and_predict(
                            matrix, labels, inner_fit, training[inner_test], regularization=regularization,
                        )
                    option = calibration_selector(calibration, training.copy())
                    threshold = option["threshold"]
                    if threshold is not None:
                        if np.isnan(threshold):
                            raise ValueError("A calibration threshold cannot be NaN")
                        test_scores = history["scores"][test]
                        present = np.isfinite(test_scores)
                        history["decisions"][test[present]] = test_scores[present] >= threshold
            history["folds"].append({
                "fold": fold,
                "train_subjects": len(np.unique(subjects[training])),
                "test_subjects": len(np.unique(subjects[test])),
                "inner_splits": inner_count if route in supported else 0,
                "fit_af_windows": int(np.sum(labels[fit] == 1)),
                "fit_non_af_windows": int(np.sum(labels[fit] == 0)),
                "held_out_finite_scores": int(np.isfinite(history["scores"][test]).sum()),
                "operating_point": option,
            })

    return {"fold_assignment": folds, "n_splits": n_splits, "routes": routes}
