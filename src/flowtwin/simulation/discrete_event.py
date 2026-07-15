from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np
import simpy


@dataclass(frozen=True)
class TaskSpec:
    name: str
    resource: str
    duration_minutes: float


@dataclass(frozen=True)
class SimulationResult:
    operation_completion_minutes: list[float]
    mean_completion_minutes: float
    p50_completion_minutes: float
    p90_completion_minutes: float
    replications: int
    evidence_type: str = "discrete_event_simulation_not_realized_saving"


class OperationSimulator:
    def __init__(
        self,
        tasks: list[TaskSpec],
        resource_capacity: dict[str, int],
        *,
        duration_noise: float = 0.15,
    ) -> None:
        if not tasks:
            raise ValueError("at least one task is required")
        self.tasks = tasks
        self.resource_capacity = resource_capacity
        self.duration_noise = duration_noise
        missing = {task.resource for task in tasks} - set(resource_capacity)
        if missing:
            raise ValueError(f"resource capacity missing for {sorted(missing)}")

    def _single(self, seed: int) -> float:
        rng = np.random.default_rng(seed)
        environment = simpy.Environment()
        resources = {
            name: simpy.Resource(environment, capacity=capacity)
            for name, capacity in self.resource_capacity.items()
        }
        completed_at = 0.0

        def operation() -> Generator[Any, Any, None]:
            nonlocal completed_at
            for task in self.tasks:
                with resources[task.resource].request() as request:
                    yield request
                    multiplier = max(0.1, rng.lognormal(mean=0, sigma=self.duration_noise))
                    yield environment.timeout(task.duration_minutes * multiplier)
            completed_at = environment.now

        environment.process(operation())
        environment.run()
        return completed_at

    def run(self, replications: int = 200, seed: int = 42) -> SimulationResult:
        if replications <= 0:
            raise ValueError("replications must be positive")
        values = [self._single(seed + index) for index in range(replications)]
        return SimulationResult(
            operation_completion_minutes=values,
            mean_completion_minutes=float(np.mean(values)),
            p50_completion_minutes=float(np.quantile(values, 0.5)),
            p90_completion_minutes=float(np.quantile(values, 0.9)),
            replications=replications,
        )
