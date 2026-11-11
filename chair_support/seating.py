"""Category evidence and competing stationary seating interpretations.

These helpers only propose and rank interpretations. They introduce no geometric
acceptance thresholds; the existing fit policy determines candidate eligibility.
"""
from copy import deepcopy
import math

import numpy as np


CATEGORY_LABELS = {"chair": 3, "sofa": 4, "stool": 7}
VERSION = "seating-v1"


def classify_locations(instances, events, labels, indices):
    """Vote only on real prediction samples within inferred support intervals."""
    labels = np.asarray(labels).reshape(len(indices), -1)
    for event in events:
        mask = np.zeros(len(indices), bool)
        for a, b in event["support"]:
            mask |= (indices >= a) & (indices < b)
        event["location_id"] = event["instance_id"]
        event["category_counts"] = {name: int(np.count_nonzero(labels[mask] == label))
                                    for name, label in CATEGORY_LABELS.items()}
        event["support_sample_count"] = int(mask.sum())
    for instance in instances:
        related = [e for e in events if e["instance_id"] == instance["instance_id"]]
        counts = {name: sum(e["category_counts"][name] for e in related) for name in CATEGORY_LABELS}
        category = min(CATEGORY_LABELS, key=lambda name: (-counts[name], CATEGORY_LABELS[name]))
        instance.update(category=category, category_counts=counts,
                        location_id=instance["instance_id"], location_ids=[instance["instance_id"]],
                        support_sample_count=sum(e["support_sample_count"] for e in related))
        for event in related:
            event["category"] = category
    return instances, events


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in _partitions(rest):
        yield [[first]] + partition
        for i in range(len(partition)):
            yield partition[:i] + [[first] + partition[i]] + partition[i + 1:]


def hypotheses(recognition, max_hypotheses=64):
    """Bounded deterministic partitions; no distance cutoff forbids a long sofa.

    The independent explanation is first, then shared-surface hints and the
    all-sofa-shared explanation. Remaining partitions enumerate every possible
    subset within the caller's search budget; hints never exclude partitions.
    Each result retains location evidence and carries an instance->anchors map.
    """
    if type(max_hypotheses) is not int or max_hypotheses < 1:
        raise ValueError("max_hypotheses must be a positive integer")
    locations = recognition.get("locations", recognition["instances"])
    by_id = {row.get("location_id", row["instance_id"]): row for row in locations}
    sofa = sorted(k for k, row in by_id.items() if row.get("category", "chair") == "sofa")
    fixed = [[k] for k in sorted(by_id) if k not in sofa]
    proposals = [[[k] for k in sofa]]
    hinted = [[k] for k in sofa]
    for hint in recognition.get("ambiguities", []):
        if hint.get("kind") != "possible_shared_seating_surface":
            continue
        members = set(hint.get("instance_ids", [])) & set(sofa)
        joined = [group for group in hinted if members.intersection(group)]
        if len(joined) > 1:
            hinted = [group for group in hinted if not members.intersection(group)]
            hinted.append(sorted(k for group in joined for k in group))
    if any(len(group) > 1 for group in hinted):
        proposals.append(hinted)
    if len(sofa) > 1:
        proposals.append([sofa])
    seen, result = set(), []
    def consume(partition):
        key = tuple(sorted(tuple(sorted(group)) for group in fixed + partition))
        if key in seen:
            return
        seen.add(key)
        report = deepcopy(recognition)
        report.update(hypothesis="stationary category-aware seating", instances=[], events=[],
                      locations=deepcopy(locations), anchor_map={}, hypothesis_id=f"seating-{len(result):03d}")
        report["grouping_search"] = "shared-surface hints order proposals; no hard merge-distance limit"
        mapping = {}
        for group in key:
            rows = [by_id[k] for k in group]
            identity = group[0] if len(group) == 1 else "sofa-" + "-".join(group)
            row = deepcopy(rows[0])
            row.update(instance_id=identity, location_ids=list(group),
                       anchor_xy_m=np.mean([r["anchor_xy_m"] for r in rows], axis=0).tolist(),
                       category_counts={name: sum(r.get("category_counts", {}).get(name, 0) for r in rows)
                                        for name in CATEGORY_LABELS},
                       support_sample_count=sum(r.get("support_sample_count", 0) for r in rows))
            row.pop("location_id", None)
            report["instances"].append(row)
            report["anchor_map"][identity] = [r["anchor_xy_m"] for r in rows]
            mapping.update({k: identity for k in group})
        for event in recognition["events"]:
            event = deepcopy(event)
            event.setdefault("location_id", event["instance_id"])
            event["instance_id"] = mapping[event["location_id"]]
            report["events"].append(event)
        for row in report["instances"]:
            row["visit_indices"] = [i for i, e in enumerate(report["events"]) if e["instance_id"] == row["instance_id"]]
        report["furniture_instance_count"] = len(report["instances"])
        result.append(report)
    for partition in proposals:
        consume(partition)
        if len(result) >= max_hypotheses:
            return result
    for partition in _partitions(sofa):
        consume(partition)
        if len(result) >= max_hypotheses:
            break
    return result


def rank_score(hypothesis, losses, instance_penalty=.001):
    """Losses are keyed by instance ID; complexity changes ranking, not validity."""
    if not math.isfinite(instance_penalty) or instance_penalty < 0:
        raise ValueError("instance_penalty must be finite and nonnegative")
    rows = hypothesis["instances"]
    weights = [max(1, r.get("support_sample_count", 0)) for r in rows]
    weighted = (sum(w * float(losses[r["instance_id"]]) for w, r in zip(weights, rows)) / sum(weights)
                if rows else 0.)
    penalty = instance_penalty * len(rows)
    return dict(weighted_fit_loss=weighted, instance_count=len(rows),
                instance_complexity_penalty=penalty, instance_penalty_weight=instance_penalty,
                total=weighted + penalty)


def heading_errors(body_forward, furniture_forward, symmetry=None):
    """Existing heading tolerance applied to equivalent functional orientations."""
    symmetry = symmetry or {"type": "none"}
    if symmetry.get("type") == "continuous":
        return np.zeros(len(np.atleast_2d(body_forward)))
    order = int(symmetry.get("order", 1)) if symmetry.get("type") == "discrete" else 1
    if order < 1:
        raise ValueError("Heading symmetry order must be positive")
    target = np.asarray(furniture_forward, float)
    angles = []
    for angle in np.arange(order) * (2 * np.pi / order):
        c, s = np.cos(angle), np.sin(angle)
        vector = np.array([c * target[0] - s * target[1], s * target[0] + c * target[1], target[2]])
        angles.append(np.degrees(np.arccos(np.clip(np.atleast_2d(body_forward) @ vector, -1., 1.))))
    return np.min(angles, axis=0)
