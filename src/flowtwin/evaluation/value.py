from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowValueAssumptions:
    cost_per_delay_hour_eur: float
    intervention_cost_eur: float
    assumed_avoidable_fraction: float


def simulated_shadow_value(
    delayed_hours: list[float],
    alerts: list[bool],
    assumptions: ShadowValueAssumptions,
) -> dict[str, float | str]:
    if len(delayed_hours) != len(alerts):
        raise ValueError("delayed_hours and alerts must have equal length")
    identified = sum(
        hours * assumptions.cost_per_delay_hour_eur * assumptions.assumed_avoidable_fraction
        for hours, alert in zip(delayed_hours, alerts, strict=True)
        if alert
    )
    interventions = sum(alerts) * assumptions.intervention_cost_eur
    return {
        "estimated_avoidable_delay_value_eur": identified,
        "estimated_intervention_cost_eur": interventions,
        "simulated_net_value_eur": identified - interventions,
        "evidence_type": "shadow_simulation_not_realized_saving",
    }
