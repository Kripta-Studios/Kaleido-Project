from __future__ import annotations

from pydantic import BaseModel


class PromotionEvidence(BaseModel):
    better_than_best_baseline: bool
    operational_metric_improved: bool
    stable_three_seeds: bool
    held_out_group_improved: bool
    correct_actions_beat_shuffled: bool | None
    calibration_acceptable: bool
    latency_acceptable: bool
    test_influenced_choice: bool


def event_jepa_promotion_gate(evidence: PromotionEvidence, action_claim: bool) -> dict[str, object]:
    checks = {
        "better_than_best_baseline": evidence.better_than_best_baseline,
        "operational_metric_improved": evidence.operational_metric_improved,
        "stable_three_seeds": evidence.stable_three_seeds,
        "held_out_group_improved": evidence.held_out_group_improved,
        "calibration_acceptable": evidence.calibration_acceptable,
        "latency_acceptable": evidence.latency_acceptable,
        "test_not_used_for_selection": not evidence.test_influenced_choice,
    }
    if action_claim:
        checks["correct_actions_beat_shuffled"] = evidence.correct_actions_beat_shuffled is True
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "language": "incremental candidate" if all(checks.values()) else "rejected_or_experimental",
    }
