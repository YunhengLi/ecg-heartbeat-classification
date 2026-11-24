"""Estimate the probability of AF time after each available score."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from .af_events import _intervals, _mask, _union
from .af_state import annotated_seconds


def time_targets(runs, stride):
    frames = []
    for run in runs:
        raw = run.scores
        if run.window_s != stride or raw.ndim != 1 or not len(raw):
            raise ValueError("Rhythm probabilities must use one fixed decision stride")
        forward = np.r_[np.nan, raw[:-1]]
        reference = _mask(run.reference_valid, raw.shape, "reference_valid")
        keep = np.isfinite(forward) & reference
        edges = run.start_s + np.arange(len(raw) + 1) * stride
        truth = annotated_seconds(edges, _union(_intervals(run.af_intervals_s))) / stride
        frames.append((forward[keep], truth[keep], np.full(np.count_nonzero(keep), run.subject_id)))
    return tuple(np.concatenate(column) for column in zip(*frames, strict=True))


def logits(scores):
    return logit(np.clip(scores, 1e-6, 1 - 1e-6))[:, None]


def fit_logit_map(scores, truth):
    if not len(truth) or not (np.any(truth > 0) and np.any(truth < 1)):
        raise ValueError("Calibration needs both AF and non-AF time on the reference clock")
    x = logits(scores)
    logistic = LogisticRegression(C=1.0, max_iter=1000)
    # A partly positive bin contributes its exact fraction to each binary outcome.
    return logistic.fit(np.concatenate((x, x)), np.repeat([0, 1], len(x)), sample_weight=np.r_[1 - truth, truth])


def summarize_probabilities(scores, truth, people):
    brier = truth * np.square(1 - scores) + (1 - truth) * np.square(scores)
    per_person = np.asarray([np.mean((scores - truth)[people == person]) for person in np.unique(people)])
    return {"mean_brier_score_by_duration": float(brier.mean()), "actual_af_time_fraction": float(truth.mean()),
            "estimated_af_time_fraction": float(scores.mean()),
            "mean_absolute_participant_burden_error": float(np.mean(np.abs(per_person))),
            "median_absolute_participant_burden_error": float(np.median(np.abs(per_person)))}
