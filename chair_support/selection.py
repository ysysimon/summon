"""Deterministic support evidence selection, independent of chair candidates.

Weights express evidence strength, not probabilities. A planar proxy selects
compatible poses; only the actual chair mesh decides geometric acceptance.
"""
import numpy as np

VERSION = 2
SLOPES = np.arange(-40, 6, dtype=float) / 100


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(np.interp(q * weights.sum(), np.cumsum(weights) - weights / 2, values))


def weighted_vector(values, weights):
    return np.array([weighted_quantile(values[:, k], weights, .5) for k in range(values.shape[1])])


def evidence_weights(chair_count, speed, vertical_speed):
    return np.minimum(np.asarray(chair_count, float) / 8, 1) / (
        1 + (np.asarray(speed, float) / .08)**2 + (np.asarray(vertical_speed, float) / .03)**2)


def support_intervals(frames):
    frames = np.asarray(frames, dtype=int)
    if not len(frames):
        return []
    cuts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1, len(frames)]
    return [(int(frames[a]), int(frames[b - 1]) + 1) for a, b in zip(cuts[:-1], cuts[1:])]


def sample_support_frames(frames, weights, fps, target_fps=15):
    """Sample source time, retaining every run's endpoints and strongest pose."""
    frames = np.asarray(frames, dtype=int)
    stride = max(1, int(round(fps / target_fps)))
    selected = set()
    for start, end in support_intervals(frames):
        selected.update(range(start, end, stride))
        selected.update((start, end - 1, start + int(np.argmax(weights[start:end]))))
    return np.array(sorted(selected), dtype=int)


def _maximum_overlap(lower, upper, weights):
    """Weighted interval sweep, with closed endpoints and deterministic ties."""
    valid = lower <= upper
    if not valid.any():
        return None
    lo, hi, w = lower[valid], upper[valid], weights[valid]
    positions, inverse = np.unique(np.r_[lo, hi], return_inverse=True)
    starts, ends = inverse[:len(lo)], inverse[len(lo):]
    start_weight = np.bincount(starts, weights=w, minlength=len(positions))
    end_weight = np.bincount(ends, weights=w, minlength=len(positions))
    totals = np.cumsum(start_weight) - np.r_[0., np.cumsum(end_weight)[:-1]]
    counts = np.cumsum(np.bincount(starts, minlength=len(positions))) - np.r_[
        0, np.cumsum(np.bincount(ends, minlength=len(positions)))[:-1]]
    # Summation order must not decide ties between slopes covering the same set.
    scores = np.round(totals, 12)
    ids = np.flatnonzero(scores == scores.max())
    index = ids[np.argmax(counts[ids])]
    mask = valid & (lower <= positions[index]) & (positions[index] <= upper)
    return mask


def select_support(frames, weights, anchors, pelvis, forward, gap_limits, penetration):
    frames = np.asarray(frames, dtype=int)
    if not len(frames):
        return frames, {"proxy_feasible": False, "reason": "no_support_evidence"}
    frame_weights = weights[frames]
    origin = weighted_vector(pelvis[frames], frame_weights)
    direction = weighted_vector(forward[frames], frame_weights)
    if np.linalg.norm(direction) < 1e-8:
        direction = forward[frames[np.argmax(frame_weights)]].copy()
    direction /= np.linalg.norm(direction)
    lower = np.full((len(frames), len(SLOPES)), -np.inf)
    upper = np.full_like(lower, np.inf)
    for region, samples in anchors.items():
        for row, (frame, _, points, areas) in enumerate(samples):
            if frame != frames[row]:
                raise ValueError("Candidate envelopes must follow source-frame order")
            heights = points[:, 2, None] - ((points - origin) @ direction)[:, None] * SLOPES
            lower[row] = np.maximum(lower[row], [weighted_quantile(heights[:, k], areas, .2)
                                                - gap_limits[region] for k in range(len(SLOPES))])
            upper[row] = np.minimum(upper[row], heights.min(axis=0) + penetration)
    best = None
    for column, slope in enumerate(SLOPES):
        mask = _maximum_overlap(lower[:, column], upper[:, column], frame_weights)
        if mask is None:
            continue
        key = (round(float(frame_weights[mask].sum()), 12), int(mask.sum()),
               -abs(float(slope)), -float(slope))
        if best is None or key > best[0]:
            best = key, column, mask
    if best is None:
        selected = frames[[np.argmax(frame_weights)]]
        return selected, dict(proxy_feasible=False, reason="no_feasible_planar_proxy",
                              fallback="strongest_valid_frame; actual curved mesh still evaluated")
    _, column, mask = best
    lo, hi = float(lower[mask, column].max()), float(upper[mask, column].min())
    return frames[mask], dict(proxy_feasible=True, slope_dz_dforward=float(SLOPES[column]),
        origin=origin.tolist(), forward=direction.tolist(), height_world_m=(lo + hi) / 2,
        height_lower_world_m=lo, height_upper_world_m=hi,
        selected_evidence_weight=float(frame_weights[mask].sum()),
        method="maximum weighted overlap of four-region seat-height intervals; geometry proxy only")
