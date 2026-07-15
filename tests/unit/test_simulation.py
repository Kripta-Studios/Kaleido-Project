from flowtwin.simulation.discrete_event import OperationSimulator, TaskSpec


def test_discrete_event_simulation_is_seeded_and_labeled() -> None:
    simulator = OperationSimulator(
        [
            TaskSpec("stage", "team", 20),
            TaskSpec("lift", "crane", 40),
        ],
        {"team": 1, "crane": 1},
    )
    left = simulator.run(replications=10, seed=7)
    right = simulator.run(replications=10, seed=7)
    assert left.operation_completion_minutes == right.operation_completion_minutes
    assert left.p90_completion_minutes >= left.p50_completion_minutes
    assert left.evidence_type == "discrete_event_simulation_not_realized_saving"
