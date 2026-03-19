"""
File-based sample locking with heartbeat timeout.

Storage: labels/locks.json (single file) + in-memory dict.
Loaded at startup, written atomically on every change via fcntl.flock.

Lock record: {user_id, locked_at, heartbeat}  (ISO-8601 UTC strings)

Rules:
  - Timeout: LOCK_TIMEOUT_SECONDS (default 300) of heartbeat inactivity
  - A user can hold only 1 lock at a time
  - Expired locks are overwritten on new acquisition
"""

import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOCKS_FILE = "labels/locks.json"

# In-memory store: sample_id → {user_id, locked_at, heartbeat}
_locks: dict[str, dict] = {}
_lock_timeout: int = 300  # seconds


def init(timeout_seconds: int = 300) -> None:
    """Load locks.json at startup. Creates file/dir if missing."""
    global _lock_timeout, _locks
    _lock_timeout = timeout_seconds

    Path("labels").mkdir(exist_ok=True)

    if os.path.exists(LOCKS_FILE):
        try:
            with open(LOCKS_FILE, "r") as f:
                _locks = json.load(f)
            logger.info(f"Loaded {len(_locks)} locks from {LOCKS_FILE}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load {LOCKS_FILE}: {e} — starting with empty locks")
            _locks = {}
    else:
        _locks = {}
        _persist()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_expired(lock_record: dict) -> bool:
    try:
        hb = datetime.fromisoformat(lock_record["heartbeat"])
        age = (datetime.now(timezone.utc) - hb).total_seconds()
        return age > _lock_timeout
    except (KeyError, ValueError):
        return True


def _persist() -> None:
    """Write _locks to disk atomically."""
    Path("labels").mkdir(exist_ok=True)
    tmp = LOCKS_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(_locks, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp, LOCKS_FILE)
    except OSError as e:
        logger.error(f"Failed to persist locks: {e}")


def acquire(sample_id: str, user_id: str) -> tuple[bool, str]:
    """
    Try to acquire a lock for sample_id by user_id.

    Returns (success, message).
    """
    # Release any existing lock held by this user (1 lock per user)
    existing_user_lock = _get_user_lock(user_id)
    if existing_user_lock and existing_user_lock != sample_id:
        _locks.pop(existing_user_lock, None)

    existing = _locks.get(sample_id)
    if existing:
        if existing["user_id"] == user_id:
            # Refresh own lock
            existing["heartbeat"] = _now()
            _persist()
            return True, "Lock refreshed"
        if not _is_expired(existing):
            return False, f"Sample locked by another user"

    # Acquire (new or expired)
    now = _now()
    _locks[sample_id] = {
        "user_id": user_id,
        "locked_at": now,
        "heartbeat": now,
    }
    _persist()
    return True, "Lock acquired"


def release(sample_id: str, user_id: str) -> tuple[bool, str]:
    """Release a lock. Only the owner can release."""
    existing = _locks.get(sample_id)
    if not existing:
        return True, "No lock held"
    if existing["user_id"] != user_id:
        return False, "Cannot release another user's lock"
    _locks.pop(sample_id)
    _persist()
    return True, "Lock released"


def heartbeat(sample_id: str, user_id: str) -> tuple[bool, str]:
    """Refresh heartbeat timestamp for an active lock."""
    existing = _locks.get(sample_id)
    if not existing:
        return False, "No lock held for this sample"
    if existing["user_id"] != user_id:
        return False, "Not the lock owner"
    existing["heartbeat"] = _now()
    _persist()
    return True, "Heartbeat updated"


def get_status(sample_id: str) -> Optional[dict]:
    """
    Return lock info dict or None if not locked / expired.
    Dict includes: user_id, locked_at, heartbeat, expired.
    """
    lock = _locks.get(sample_id)
    if not lock:
        return None
    expired = _is_expired(lock)
    return {
        "user_id": lock["user_id"],
        "locked_at": lock["locked_at"],
        "heartbeat": lock["heartbeat"],
        "expired": expired,
    }


def is_locked(sample_id: str) -> bool:
    """True if sample has an active (non-expired) lock."""
    lock = _locks.get(sample_id)
    return lock is not None and not _is_expired(lock)


def is_locked_by(sample_id: str, user_id: str) -> bool:
    lock = _locks.get(sample_id)
    return (
        lock is not None
        and lock["user_id"] == user_id
        and not _is_expired(lock)
    )


def _get_user_lock(user_id: str) -> Optional[str]:
    """Return sample_id currently locked by user_id (active only), or None."""
    for sid, lock in _locks.items():
        if lock["user_id"] == user_id and not _is_expired(lock):
            return sid
    return None


def cleanup_expired() -> int:
    """Remove all expired locks. Returns count removed."""
    expired_keys = [k for k, v in _locks.items() if _is_expired(v)]
    for k in expired_keys:
        _locks.pop(k)
    if expired_keys:
        _persist()
    return len(expired_keys)
