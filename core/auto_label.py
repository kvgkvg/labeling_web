"""
Auto-labeling rules applied at save time.

Computes deterministic fields from user inputs + sample data:
  bbox_correct          — IoU(user GT bboxes, model extracted_bbox) > IOU_THRESHOLD
  bbox_answer_changed   — bbox_answer != mcq_answer
  reasoning_correct     — user made no edit to chain-of-thought (edited == original)
  reason_answer_changed — reason_answer != mcq_answer
"""

import os
from typing import Optional

from core.iou import compute_iou

IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.8"))


def compute_bbox_correct(
    model_bboxes: list[dict],
    groundtruth_bboxes: list[dict],
    threshold: Optional[float] = None,
) -> Optional[bool]:
    """
    Compare model bboxes to user-drawn GT bboxes via IoU.

    Returns:
        None   — both lists empty (no label, show as N/A)
        False  — one list empty (model missed or user drew extra)
        bool   — IoU > threshold
    """
    thr = threshold if threshold is not None else IOU_THRESHOLD
    iou = compute_iou(model_bboxes, groundtruth_bboxes)
    if iou is None:
        return None
    return iou > thr


def compute_auto_labels(
    sample: dict,
    groundtruth_bboxes: list[dict],
    edited_visual_reasoning: str,
    edited_reasoning: str,
) -> dict:
    """
    Compute all auto-labeled fields for a sample submission.

    Args:
        sample:                   The sample dict from data_loader
        groundtruth_bboxes:       User-drawn GT bboxes [{x1,y1,x2,y2,label}, ...]
        edited_visual_reasoning:  Current value of visual reasoning textarea
        edited_reasoning:         Current value of reasoning textarea

    Returns dict with:
        iou_score, bbox_correct,
        bbox_answer_changed, visual_reasoning_correct,
        reasoning_correct, reason_answer_changed
    """
    model_bboxes = sample.get("extracted_bbox") or []
    mcq_answer = (sample.get("mcq_answer") or "").strip().upper()
    bbox_answer = (sample.get("bbox_answer") or "").strip().upper()
    reason_answer = (sample.get("reason_answer") or "").strip().upper()
    original_visual_reasoning = sample.get("visual_reasoning") or ""
    original_reasoning = sample.get("reasoning") or ""

    # IoU
    iou = compute_iou(model_bboxes, groundtruth_bboxes)

    # bbox_correct
    if iou is None:
        bbox_correct = None
    else:
        bbox_correct = iou > IOU_THRESHOLD

    # answer change flags
    bbox_answer_changed = (bbox_answer != mcq_answer) if (bbox_answer and mcq_answer) else False
    reason_answer_changed = (reason_answer != mcq_answer) if (reason_answer and mcq_answer) else False

    # reasoning correctness — True if user made no edits
    visual_reasoning_correct = (edited_visual_reasoning.strip() == original_visual_reasoning.strip())
    reasoning_correct = (edited_reasoning.strip() == original_reasoning.strip())

    return {
        "iou_score": iou,
        "bbox_correct": bbox_correct,
        "bbox_answer_changed": bbox_answer_changed,
        "visual_reasoning_correct": visual_reasoning_correct,
        "reasoning_correct": reasoning_correct,
        "reason_answer_changed": reason_answer_changed,
    }
