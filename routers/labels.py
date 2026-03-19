"""
Label save/load endpoints.

GET  /api/labels/{sample_id}   Get existing label (or 404)
POST /api/labels/{sample_id}   Save label (validates lock ownership)
"""

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import data_loader, lock_manager
from core.auto_label import compute_auto_labels

logger = logging.getLogger(__name__)
router = APIRouter()

LABELS_DIR = "labels"


# ---------------------------------------------------------------------------
# Pydantic input model
# ---------------------------------------------------------------------------

class BboxDict(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""


class SaveLabelRequest(BaseModel):
    user_id: str

    # Bbox layer
    groundtruth_bboxes: list[BboxDict] = []

    # Reasoning layers
    edited_visual_reasoning: str = ""
    edited_reasoning: str = ""

    # MCQ layer (groundtruth override by labeler if they change it)
    groundtruth_answer: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _labels_path(dataset: str) -> str:
    return os.path.join(LABELS_DIR, f"{dataset}.json")


def _load_dataset_labels(dataset: str) -> dict:
    path = _labels_path(dataset)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load labels for {dataset}: {e}")
        return {}


def _save_dataset_label(dataset: str, row_idx: int, label: dict) -> None:
    """Atomically update a single label entry in the dataset's JSON file."""
    Path(LABELS_DIR).mkdir(exist_ok=True)
    path = _labels_path(dataset)

    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            data = json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, OSError):
            data = {}

        data[str(row_idx)] = label

        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=2, ensure_ascii=False)
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/labels/{sample_id}")
def get_label(sample_id: str):
    sample = data_loader.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    dataset = sample["dataset"]
    row_idx = sample["row_idx"]
    labels = _load_dataset_labels(dataset)
    label = labels.get(str(row_idx))

    if label is None:
        raise HTTPException(status_code=404, detail="Label not found")
    return {"label": label}


@router.post("/labels/{sample_id}")
def save_label(sample_id: str, body: SaveLabelRequest):
    sample = data_loader.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    # Validate lock ownership
    if not lock_manager.is_locked_by(sample_id, body.user_id):
        raise HTTPException(
            status_code=403,
            detail="You must hold the lock for this sample to save a label",
        )

    gt_bboxes = [b.model_dump() for b in body.groundtruth_bboxes]

    # Compute auto-labels
    auto = compute_auto_labels(
        sample=sample,
        groundtruth_bboxes=gt_bboxes,
        edited_visual_reasoning=body.edited_visual_reasoning,
        edited_reasoning=body.edited_reasoning,
    )

    now = datetime.now(timezone.utc).isoformat()

    label = {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "row_idx": sample["row_idx"],
        "labeled_by": body.user_id,
        "labeled_at": now,

        # Sample content for traceability
        "question": sample.get("question"),
        "choices": sample.get("choices"),
        "images": sample.get("images"),
        "category": sample.get("category"),

        # Status gates
        "bbox_status": sample.get("bbox_status"),
        "reasoning_status": sample.get("reasoning_status"),
        "answer_status": sample.get("answer_status"),

        # Bbox labels
        "bbox_labels": {
            "model_bboxes": sample.get("extracted_bbox") or [],
            "groundtruth_bboxes": gt_bboxes,
            "iou_score": auto["iou_score"],
            "bbox_correct": auto["bbox_correct"],
            "bbox_answer_changed": auto["bbox_answer_changed"],
        },

        # Visual reasoning labels
        "bbox_reasoning_labels": {
            "original_visual_reasoning": sample.get("visual_reasoning") or "",
            "edited_visual_reasoning": body.edited_visual_reasoning,
            "visual_reasoning_correct": auto["visual_reasoning_correct"],
        },

        # Reasoning labels
        "reasoning_labels": {
            "original_reasoning": sample.get("reasoning") or "",
            "edited_reasoning": body.edited_reasoning,
            "reasoning_correct": auto["reasoning_correct"],
            "reason_answer_changed": auto["reason_answer_changed"],
        },

        # MCQ labels
        "mcq_labels": {
            "groundtruth": sample.get("answer"),
            "mcq_answer": sample.get("mcq_answer"),
            "groundtruth_override": body.groundtruth_answer,
        },
    }

    _save_dataset_label(sample["dataset"], sample["row_idx"], label)

    # Release lock after successful save
    lock_manager.release(sample_id, body.user_id)

    return {"status": "saved", "label": label}
