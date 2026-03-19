"""
IoU calculation in normalized 0-1000 coordinate space.

Greedy matching: sort predicted boxes by area descending, match each to the
nearest unmatched GT box by maximum IoU. Average over matched pairs.
Unmatched boxes contribute IoU = 0.

Edge cases:
  - Both lists empty  → None  (no label, display as N/A)
  - One list empty    → 0.0
"""

from typing import Optional


def _box_area(b: dict) -> float:
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])


def _single_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = _box_area(a) + _box_area(b) - inter
    if union == 0:
        return 0.0
    return inter / union


def compute_iou(predicted: list[dict], groundtruth: list[dict]) -> Optional[float]:
    """
    Compute mean greedy-matched IoU between two bbox lists.

    Args:
        predicted:   list of {x1, y1, x2, y2, label} (model output)
        groundtruth: list of {x1, y1, x2, y2, label} (user annotated)

    Returns:
        None  if both lists are empty
        0.0   if one list is empty
        float mean IoU in [0, 1] otherwise
    """
    if not predicted and not groundtruth:
        return None
    if not predicted or not groundtruth:
        return 0.0

    # Sort predicted by area descending
    sorted_pred = sorted(predicted, key=_box_area, reverse=True)
    unmatched_gt = list(range(len(groundtruth)))
    total_iou = 0.0
    matched = 0

    for pred in sorted_pred:
        if not unmatched_gt:
            break
        best_iou = -1.0
        best_idx = -1
        for gi in unmatched_gt:
            iou = _single_iou(pred, groundtruth[gi])
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_idx >= 0:
            total_iou += best_iou
            matched += 1
            unmatched_gt.remove(best_idx)

    total_boxes = max(len(predicted), len(groundtruth))
    return total_iou / total_boxes
