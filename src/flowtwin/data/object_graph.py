from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field

from flowtwin.data.contracts import EventObjectRelation, ObjectReference, ObjectRelation


class ObjectGraphReport(BaseModel):
    passed: bool
    object_count: int
    event_count: int
    event_object_edges: int
    object_object_edges: int
    objects_by_type: dict[str, int]
    orphan_event_ids: list[str] = Field(default_factory=list)
    missing_object_ids: list[str] = Field(default_factory=list)


class ObjectGraph:
    def __init__(
        self,
        objects: Iterable[ObjectReference],
        event_ids: Iterable[str],
        event_object: Iterable[EventObjectRelation],
        object_object: Iterable[ObjectRelation] = (),
    ) -> None:
        self.objects = {item.object_id: item for item in objects}
        self.event_ids = set(event_ids)
        self.event_object = list(event_object)
        self.object_object = list(object_object)

    def validate(self) -> ObjectGraphReport:
        attached_events: set[str] = set()
        missing_objects: set[str] = set()
        for relation in self.event_object:
            attached_events.add(relation.event_id)
            if relation.object_id not in self.objects:
                missing_objects.add(relation.object_id)
        for object_relation in self.object_object:
            if object_relation.source_object_id not in self.objects:
                missing_objects.add(object_relation.source_object_id)
            if object_relation.target_object_id not in self.objects:
                missing_objects.add(object_relation.target_object_id)
        orphan_events = self.event_ids - attached_events
        return ObjectGraphReport(
            passed=not missing_objects and not orphan_events,
            object_count=len(self.objects),
            event_count=len(self.event_ids),
            event_object_edges=len(self.event_object),
            object_object_edges=len(self.object_object),
            objects_by_type=dict(Counter(item.object_type for item in self.objects.values())),
            orphan_event_ids=sorted(orphan_events),
            missing_object_ids=sorted(missing_objects),
        )

    def event_neighbors(self) -> dict[str, set[str]]:
        object_events: dict[str, set[str]] = defaultdict(set)
        for relation in self.event_object:
            object_events[relation.object_id].add(relation.event_id)
        neighbors: dict[str, set[str]] = defaultdict(set)
        for event_ids in object_events.values():
            for event_id in event_ids:
                neighbors[event_id].update(event_ids - {event_id})
        return dict(neighbors)
