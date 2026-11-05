"""Schema-driven event preprocessing for the unified V26 protocol."""
import collections
import gzip
import json
import xml.etree.ElementTree as ET

import numpy as np


def normalize_ties(events):
    """Normalize observed timestamp ties without enlarging the 50-event window."""
    events = list(events[:50])
    output = []
    start = 0
    while start < len(events):
        end = start + 1
        timestamp = events[start].get("time:timestamp")
        while end < len(events) and timestamp is not None and events[end].get("time:timestamp") == timestamp:
            end += 1
        group = events[start:end]
        if all(event.get("action_code") for event in group):
            group = sorted(group, key=lambda event: event["action_code"])
        output.extend(group)
        start = end
    assert len(output) == len(events)
    assert collections.Counter(event["concept:name"] for event in output) == collections.Counter(
        event["concept:name"] for event in events
    )
    return output


def read_events(path):
    """Read complete event dictionaries for each trace using the common order rule."""
    cases = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        for _, trace in ET.iterparse(handle, events=("end",)):
            if trace.tag.rsplit("}", 1)[-1] != "trace":
                continue
            name = None
            events = []
            for child in trace:
                if child.tag.rsplit("}", 1)[-1] == "event":
                    event = {item.get("key"): item.get("value") for item in child}
                    assert event.get("concept:name")
                    events.append(event)
                elif child.get("key") == "concept:name":
                    name = child.get("value")
            assert events
            cases["%d:%s" % (len(cases), name)] = normalize_ties(events)
            trace.clear()
    return cases


def choose_representation(train_sequences):
    """Choose representation using only the current training split schema."""
    lifecycle = [
        str(event.get("lifecycle:transition", "")).strip()
        for sequence in train_sequences
        for event in sequence
    ]
    states = sorted(set(value for value in lifecycle if value))
    complete = bool(lifecycle) and all(lifecycle)
    compound = complete and len(states) >= 2
    audit = dict(
        source="training split only",
        training_events=len(lifecycle),
        lifecycle_coverage=float(sum(bool(value) for value in lifecycle) / max(1, len(lifecycle))),
        distinct_lifecycle_states=len(states),
        lifecycle_states=states,
        decision="activity_lifecycle" if compound else "activity",
    )
    return compound, audit


def token(event, compound):
    activity = event["concept:name"]
    if not compound:
        return activity
    return json.dumps(
        [activity, str(event["lifecycle:transition"]).strip()],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def vocabulary(train_sequences, compound):
    return {
        value: index + 4
        for index, value in enumerate(sorted({token(event, compound) for sequence in train_sequences for event in sequence}))
    }


def encode(event_sequences, vocab, compound):
    output = np.zeros((len(event_sequences), 52), np.int64)
    for index, sequence in enumerate(event_sequences):
        assert 0 < len(sequence) <= 50
        encoded = [1] + [vocab.get(token(event, compound), 3) for event in sequence] + [2]
        output[index, : len(encoded)] = encoded
    return output


def replay(original, realized, training_events, rng):
    """Replay fixed activity-level perturbations while retaining observed event attributes."""
    current = list(original)
    operations = []
    for item in realized:
        operation = item["operation"]
        kind = operation["type"]
        position = operation["position"]
        extra = {}
        assert 0 <= position < len(current)
        if kind == "skip":
            assert len(current) > 1
            current.pop(position)
        elif kind == "insert":
            assert len(current) < 50
            candidates = training_events[operation["value"]]
            assert candidates
            event = candidates[int(rng.integers(len(candidates)))]
            current.insert(position, event)
            extra["inserted_event"] = event
        elif kind == "rework":
            assert len(current) < 50
            current.insert(position, current[position])
        elif kind in ("early", "late"):
            target = operation["target"]
            assert 0 <= target < len(current)
            assert (target < position if kind == "early" else target > position)
            current.insert(target, current.pop(position))
        else:
            raise ValueError(kind)
        operations.append(dict(operation=operation, **extra))
    return current, operations
