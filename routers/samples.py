"""
Sample management endpoints.

GET  /api/samples                          List all samples with status
GET  /api/samples/next                     Next unlocked, unlabeled sample for user
GET  /api/samples/{sample_id}              Full sample data
POST /api/samples/{sample_id}/lock         Acquire lock
POST /api/samples/{sample_id}/unlock       Release lock
POST /api/samples/{sample_id}/heartbeat    Refresh heartbeat
GET  /api/samples/{sample_id}/lock-status  Check lock status
GET  /api/progress                         Per-dataset progress summary
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core import data_loader, lock_manager
from routers.labels import _load_dataset_labels

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LockRequest(BaseModel):
    user_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_summary(sample: dict, labeled_ids: set[str]) -> dict:
    """Return a lightweight summary dict for list endpoints."""
    sid = sample["id"]
    lock_info = lock_manager.get_status(sid)
    return {
        "id": sid,
        "dataset": sample["dataset"],
        "row_idx": sample["row_idx"],
        "category": sample.get("category"),
        "is_valid": sample["is_valid"],
        "is_labeled": sid in labeled_ids,
        "lock": lock_info,
    }


def _build_labeled_set() -> set[str]:
    """Collect all labeled sample IDs across all dataset label files."""
    from core.data_loader import list_samples
    datasets = {s["dataset"] for s in list_samples().values()}
    labeled: set[str] = set()
    for ds in datasets:
        ds_labels = _load_dataset_labels(ds)
        for row_idx_str, label in ds_labels.items():
            labeled.add(label.get("id", f"{ds}_{row_idx_str}"))
    return labeled


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/samples")
def list_all_samples(
    dataset: str = Query(None, description="Filter by dataset name"),
    valid_only: bool = Query(False, description="Only return valid samples"),
):
    samples = data_loader.list_samples()
    labeled_ids = _build_labeled_set()

    result = []
    for sample in samples.values():
        if dataset and sample["dataset"] != dataset:
            continue
        if valid_only and not sample["is_valid"]:
            continue
        result.append(_sample_summary(sample, labeled_ids))

    return {"total": len(result), "samples": result}


@router.get("/samples/next")
def get_next_sample(
    user_id: str = Query(..., description="Labeler user ID"),
    dataset: str = Query(None, description="Restrict to a specific dataset"),
    exclude: str = Query(None, description="Comma-separated sample IDs to skip"),
):
    """
    Return next valid, unlabeled, unlocked sample for user_id.
    Skips samples already labeled by anyone, or locked by another user.
    """
    samples = data_loader.list_samples()
    labeled_ids = _build_labeled_set()
    excluded_ids = set(exclude.split(',')) if exclude else set()

    for sample in samples.values():
        if not sample["is_valid"]:
            continue
        if dataset and sample["dataset"] != dataset:
            continue

        sid = sample["id"]
        if sid in labeled_ids:
            continue
        if sid in excluded_ids:
            continue

        # Check lock
        lock = lock_manager.get_status(sid)
        if lock and not lock["expired"] and lock["user_id"] != user_id:
            continue

        # Candidate — acquire lock
        ok, msg = lock_manager.acquire(sid, user_id)
        if ok:
            return {"sample": _build_full_sample(sample)}
        # If lock failed (race), skip
        continue

    return {"sample": None, "message": "No available samples"}


@router.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    sample = data_loader.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    return {"sample": _build_full_sample(sample)}


@router.post("/samples/{sample_id}/lock")
def lock_sample(sample_id: str, body: LockRequest):
    sample = data_loader.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    ok, msg = lock_manager.acquire(sample_id, body.user_id)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"status": "locked", "message": msg}


@router.post("/samples/{sample_id}/unlock")
def unlock_sample(sample_id: str, body: LockRequest):
    sample = data_loader.get_sample(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")

    ok, msg = lock_manager.release(sample_id, body.user_id)
    if not ok:
        raise HTTPException(status_code=403, detail=msg)
    return {"status": "unlocked", "message": msg}


@router.post("/samples/{sample_id}/heartbeat")
def sample_heartbeat(sample_id: str, body: LockRequest):
    ok, msg = lock_manager.heartbeat(sample_id, body.user_id)
    if not ok:
        raise HTTPException(status_code=403, detail=msg)
    return {"status": "ok", "message": msg}


@router.get("/samples/{sample_id}/lock-status")
def get_lock_status(sample_id: str):
    lock = lock_manager.get_status(sample_id)
    if lock is None:
        return {"locked": False}
    return {"locked": not lock["expired"], **lock}


@router.get("/progress")
def get_progress():
    """Per-dataset labeled / total counts."""
    samples = data_loader.list_samples()

    # Group valid samples by dataset
    dataset_totals: dict[str, int] = {}
    for s in samples.values():
        if s["is_valid"]:
            dataset_totals[s["dataset"]] = dataset_totals.get(s["dataset"], 0) + 1

    result = {}
    for ds, total in dataset_totals.items():
        ds_labels = _load_dataset_labels(ds)
        result[ds] = {
            "total": total,
            "labeled": len(ds_labels),
            "remaining": total - len(ds_labels),
        }
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_full_sample(sample: dict) -> dict:
    """Return the full sample dict suitable for API response."""
    sid = sample["id"]
    lock = lock_manager.get_status(sid)
    return {
        **sample,
        # extracted_bbox is already parsed list of dicts — serialisable
        "lock": lock,
    }
