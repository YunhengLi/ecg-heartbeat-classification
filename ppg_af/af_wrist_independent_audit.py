"""Audit a published frozen template ECG test without selecting test parameters."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .af_wrist_failure_inputs import load_input_evidence
from .af_wrist_failure_study import audit_route
from .af_wrist_holdout import _predict, _write_json
from .af_wrist_independent_false_audit import summarize_false_alerts
from .af_wrist_independent_miss_audit import summarize_missed_episodes
from .af_wrist_template_independent import ROUTE, load_independent_inputs
from .af_wrist_vote_study import _held_out_replay, _IMMEDIATE


def replay_published(lock, protocol, independent_features, stride_raw, stride_features, coverage, published):
    """Reconcile the original endpoint counts before summarizing event errors."""
    grid, frozen, available, _, roster, _, _ = load_independent_inputs(
        lock, protocol, independent_features, stride_raw, stride_features, coverage,
    )
    if set(grid.subjects) != roster or len(roster) != published["evaluated_independent_participants"]:
        raise ValueError("The audit must use the complete published independent roster")
    scores = _predict(grid.table[ROUTE], available, frozen["model"])
    cutoff = frozen["cutoff"]
    cutoffs = np.full(len(scores), cutoff)
    decisions = np.where(np.isfinite(scores), scores >= cutoff, -1).astype(np.int8)
    evidence = load_input_evidence(grid, stride_raw, stride_features)
    summary, cases = audit_route(grid, 1, scores, cutoffs, decisions, evidence, published["independent"])
    _, reference, _ = _held_out_replay(grid, scores, cutoffs, _IMMEDIATE)
    complete = cases["complete_episodes"]
    return {
        "scope": "Post-hoc error audit of the fixed independent template ECG result",
        "independent_participants": len(roster),
        "complete_reference_episodes": summary["complete_episodes"],
        "complete_episodes_detected": summary["detected"],
        "partial_reference_episodes": summary["partial_episodes"],
        "wrist_false_alerts": summary["false_alerts"]["starts"],
        "miss_audit": summarize_missed_episodes(complete, reference),
        "false_alert_audit": summarize_false_alerts(grid, scores, cutoff, cases["false_alerts"], evidence,
                                                   episodes=complete),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "protocol-lock", "independent-features", "stride-raw", "stride-features",
                 "coverage-report", "published-results", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock_bytes = args.lock.read_bytes()
        protocol_bytes = args.protocol_lock.read_bytes()
        coverage_bytes = args.coverage_report.read_bytes()
        published_bytes = args.published_results.read_bytes()
        lock, published = json.loads(lock_bytes), json.loads(published_bytes)
        coverage_digests = {sha256(data).hexdigest() for data in (coverage_bytes, coverage_bytes.replace(b"\r\n", b"\n"))}
        if (sha256(lock_bytes).hexdigest() != published["model_lock_sha256"]
                or sha256(protocol_bytes).hexdigest() != lock["source_protocol_sha256"]
                or published["coverage_report_sha256"] not in coverage_digests):
            raise ValueError("The audit must preserve the published model, protocol, and coverage receipt")
        report = replay_published(lock, json.loads(protocol_bytes), args.independent_features,
                                  args.stride_raw, args.stride_features, json.loads(coverage_bytes), published)
        report["published_result_sha256"] = sha256(published_bytes.replace(b"\r\n", b"\n")).hexdigest()
        _write_json(args.output, report)
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"Independent template ECG audit failed: {error}\n")
    print(f"Audited the published results for {report['independent_participants']} independent participants")


if __name__ == "__main__":
    main()
