from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from flowtwin.data.object_graph import ObjectGraph


def graph_prefix_features(graph: ObjectGraph, ordered_event_ids: list[str]) -> np.ndarray:
    """Deterministic object-neighborhood features for a non-neural graph floor."""
    neighbors = graph.event_neighbors()
    object_types = {key: value.object_type for key, value in graph.objects.items()}
    event_objects: dict[str, list[str]] = defaultdict(list)
    for relation in graph.event_object:
        event_objects[relation.event_id].append(relation.object_id)
    rows: list[list[float]] = []
    seen_objects: set[str] = set()
    seen_types: Counter[str] = Counter()
    for index, event_id in enumerate(ordered_event_ids, start=1):
        linked = event_objects.get(event_id, [])
        seen_objects.update(linked)
        seen_types.update(
            object_types[object_id] for object_id in linked if object_id in object_types
        )
        rows.append(
            [
                float(index),
                float(len(linked)),
                float(len(seen_objects)),
                float(len(neighbors.get(event_id, set()))),
                float(len(seen_types)),
                float(max(seen_types.values(), default=0)),
            ]
        )
    return np.asarray(rows, dtype=float)
