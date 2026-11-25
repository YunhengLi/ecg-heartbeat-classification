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


def _within_rhythm_participant_weights(labels, subjects):
    """Give equal total weight to each participant within a training rhythm."""
    weights = np.empty(len(labels), dtype=float)
    for label in (0, 1):
        rows = labels == label
        _, indices, counts = np.unique(subjects[rows], return_inverse=True, return_counts=True)
        weights[rows] = len(labels) / (2 * len(counts) * counts[indices])
    return weights


def _fit_and_predict(features, labels, train_rows, score_rows, *, regularization, weighted_subjects=None):
    predictions = np.full(len(score_rows), np.nan)
    available = np.isfinite(features[score_rows]).all(axis=1)
    if len(train_rows) and np.any(available):
        class_weight = "balanced" if weighted_subjects is None else None
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, class_weight=class_weight, max_iter=1000),
        )
        training_labels = labels[train_rows]
        fit_options = {}
        if weighted_subjects is not None:
            fit_options["logisticregression__sample_weight"] = _within_rhythm_participant_weights(
                training_labels, weighted_subjects[train_rows],
            )
        classifier.fit(features[train_rows], training_labels, **fit_options)
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
    participant_weighting=False, supplemental_fit_labels=None, fold_feature_transform: Callable | None = None,
    fold_predictor: Callable | None = None,
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
    With ``participant_weighting``, the logistic fit balances participant
    contributions within each rhythm using that fit's training rows only.
    Optional ``supplemental_fit_labels`` add labels on previously unlabeled
    rows for fitting. Split assignments and split support use the original
    labels. Supplemental rows can fit a model only within its training group.
    Optional ``fold_feature_transform(matrix, fit_rows, score_rows)`` returns
    an aligned matrix for one fit or inner calibration fit. It receives the
    original read-only route matrix and the exact selected row indices. It
    must preserve which fitting and scoring rows have finite features.
    Optional ``fold_predictor(train_features, train_labels, score_features)``
    replaces logistic regression. It receives only training labels and finite
    input rows, and returns one AF probability per score row.
    """
    if not isfinite(regularization) or regularization <= 0:
        raise ValueError("Logistic regression C must be positive and finite")
    labels = np.asarray(labels)
    subjects = np.asarray(subjects, dtype=str)
    if (labels.ndim != 1 or not len(labels) or not np.isin(labels, (-1, 0, 1)).all()
            or subjects.shape != labels.shape or np.any(subjects == "")):
        raise ValueError("Expected aligned participant IDs and labels in {-1, 0, 1}")
    labels = labels.astype(int)
    fit_options = {"regularization": regularization}
    if participant_weighting:
        fit_options["weighted_subjects"] = subjects
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
    fitting_labels, fitting_masks = fit_labels, masks
    if supplemental_fit_labels is not None:
        supplemental = np.asarray(supplemental_fit_labels)
        if (supplemental.shape != labels.shape or not np.isin(supplemental, (-1, 0, 1)).all()
                or np.any((supplemental >= 0) & (labels >= 0))):
            raise ValueError("Supplemental labels must align to previously unlabeled training rows")
        fitting_labels = np.where(supplemental >= 0, supplemental, fit_labels).astype(int)
        fitting_masks = {route: mask | ((supplemental >= 0) & np.isfinite(matrices[route]).all(axis=1))
                         for route, mask in masks.items()}

    def predict(matrix, train_rows, score_rows):
        if fold_feature_transform is not None:
            original = matrix.view()
            original.flags.writeable = False
            rows = np.r_[train_rows, score_rows]
            transformed = np.asarray(fold_feature_transform(original, train_rows.copy(), score_rows.copy()), dtype=float)
            if (transformed.shape != matrix.shape
                    or not np.array_equal(np.isfinite(transformed[rows]).all(axis=1), np.isfinite(matrix[rows]).all(axis=1))):
                raise ValueError("Fold feature transforms must preserve matrix shape and finite fit and score rows")
            matrix = transformed
        if fold_predictor is not None:
            available = np.isfinite(matrix[score_rows]).all(axis=1)
            predictions = np.full(len(score_rows), np.nan)
            if available.any():
                selected = np.asarray(fold_predictor(matrix[train_rows], fitting_labels[train_rows], matrix[score_rows[available]]))
                if selected.shape != (int(available.sum()),) or not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
                    raise ValueError("A custom fold predictor must return finite probabilities on every supported score row")
                predictions[available] = selected
            return predictions
        return _fit_and_predict(matrix, fitting_labels, train_rows, score_rows, **fit_options)

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
            baseline_fit = training[masks[route][training]]
            fit = training[fitting_masks[route][training]]
            option = {"threshold": None, "status": "not_calibrated"}
            if not _both_classes(fit_labels[baseline_fit]):
                option["status"] = "insufficient_training_labels"
            else:
                history["scores"][test] = predict(matrix, fit, test)
                if calibration_selector is not None:
                    option["status"] = "insufficient_calibration_training_labels"
                if route in supported:
                    calibration = np.full(len(training), np.nan)
                    for inner_fold in range(inner_count):
                        inner_fit = training[(inner != inner_fold) & fitting_masks[route][training]]
                        inner_test = inner == inner_fold
                        calibration[inner_test] = predict(matrix, inner_fit, training[inner_test])
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
                "fit_af_windows": int(np.sum(fitting_labels[fit] == 1)),
                "fit_non_af_windows": int(np.sum(fitting_labels[fit] == 0)),
                "held_out_finite_scores": int(np.isfinite(history["scores"][test]).sum()),
                "operating_point": option,
            })

    return {"fold_assignment": folds, "n_splits": n_splits, "routes": routes}
