from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from flowtwin.data.roles import FUTURE_PATTERNS, ColumnRole, FieldClassification
from flowtwin.data.splits import Partition


class LeakageSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LeakageFinding(BaseModel):
    code: str
    severity: LeakageSeverity
    message: str
    fields: list[str] = Field(default_factory=list)


class LeakageReport(BaseModel):
    passed: bool
    unsafe_debug: bool
    watermark: str | None
    findings: list[LeakageFinding]

    def require_passed(self) -> None:
        if not self.passed:
            raise RuntimeError("leakage audit failed closed; inspect leakage_report.json")


def run_leakage_audit(
    classification: FieldClassification,
    feature_fields: Iterable[str],
    operation_partitions: Iterable[tuple[str, Partition]] = (),
    prefix_cutoffs: Iterable[tuple[str, datetime, datetime]] = (),
    unsafe_debug: bool = False,
) -> LeakageReport:
    findings: list[LeakageFinding] = []
    features = set(feature_fields)
    prohibited_roles = classification.fields(ColumnRole.OUTCOME) | classification.fields(
        ColumnRole.FORBIDDEN
    )
    role_overlap = features & prohibited_roles
    if role_overlap:
        findings.append(
            LeakageFinding(
                code="outcome_or_forbidden_feature",
                severity=LeakageSeverity.ERROR,
                message="feature set contains outcome or explicitly forbidden fields",
                fields=sorted(role_overlap),
            )
        )
    suspicious = sorted(
        field for field in features if any(pattern.search(field) for pattern in FUTURE_PATTERNS)
    )
    if suspicious:
        findings.append(
            LeakageFinding(
                code="future_name_pattern",
                severity=LeakageSeverity.ERROR,
                message="feature names suggest future/outcome information",
                fields=suspicious,
            )
        )
    observed_partitions: dict[str, set[Partition]] = {}
    for operation_id, partition in operation_partitions:
        observed_partitions.setdefault(operation_id, set()).add(partition)
    crossing = sorted(key for key, partitions in observed_partitions.items() if len(partitions) > 1)
    if crossing:
        findings.append(
            LeakageFinding(
                code="operation_crosses_split",
                severity=LeakageSeverity.ERROR,
                message=f"{len(crossing)} operations occur in multiple partitions",
                fields=crossing[:20],
            )
        )
    cutoff_violations = [
        operation_id for operation_id, event_time, cutoff in prefix_cutoffs if event_time > cutoff
    ]
    if cutoff_violations:
        findings.append(
            LeakageFinding(
                code="event_after_prediction_cutoff",
                severity=LeakageSeverity.ERROR,
                message=f"{len(cutoff_violations)} prefixes contain future events",
                fields=sorted(set(cutoff_violations))[:20],
            )
        )
    errors = any(item.severity == LeakageSeverity.ERROR for item in findings)
    if errors and unsafe_debug:
        findings.append(
            LeakageFinding(
                code="unsafe_debug_enabled",
                severity=LeakageSeverity.WARNING,
                message="artifacts are watermarked smoke-only and cannot enter reports",
            )
        )
    return LeakageReport(
        passed=not errors,
        unsafe_debug=unsafe_debug,
        watermark="UNSAFE_DEBUG_NOT_CLAIM_ELIGIBLE" if unsafe_debug else None,
        findings=findings,
    )
