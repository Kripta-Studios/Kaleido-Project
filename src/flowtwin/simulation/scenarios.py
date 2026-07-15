from __future__ import annotations

from dataclasses import dataclass

from flowtwin.simulation.discrete_event import OperationSimulator, SimulationResult


@dataclass(frozen=True)
class ApprovedScenario:
    scenario_id: str
    label: str
    approved_action: str
    simulator: OperationSimulator


def compare_scenarios(
    scenarios: list[ApprovedScenario], replications: int = 200, seed: int = 42
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for index, scenario in enumerate(scenarios):
        result: SimulationResult = scenario.simulator.run(
            replications=replications, seed=seed + 1000 * index
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "label": scenario.label,
                "approved_action": scenario.approved_action,
                "p50_completion_minutes": result.p50_completion_minutes,
                "p90_completion_minutes": result.p90_completion_minutes,
                "replications": result.replications,
                "evidence_type": result.evidence_type,
            }
        )
    return sorted(rows, key=lambda row: float(row["p50_completion_minutes"]))
