"""
Build a {filename: abs_path} cache by recursively scanning IMAGE_BASE_DIR.
Lookup is by basename only — dataset subdirectory structure is transparent.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Global caches
_cache: dict[str, str] = {}       # basename (with ext) → abs_path
_stem_cache: dict[str, str] = {}  # stem (no ext) → abs_path


def build_cache(image_base_dir: str) -> dict[str, str]:
    """
    Recursively scan image_base_dir and build filename→abs_path mapping.
    Also builds a stem-only fallback cache for datasets that omit extensions.
    Call once at server startup. Returns the cache dict.
    """
    global _cache, _stem_cache
    _cache = {}
    _stem_cache = {}

    base = Path(os.path.expanduser(image_base_dir))
    if not base.exists():
        logger.warning(f"IMAGE_BASE_DIR does not exist: {base} — image cache will be empty")
        return _cache

    count = 0
    for path in base.rglob("*"):
        if path.is_file():
            name = path.name
            stem = path.stem
            abs_str = str(path.resolve())
            if name not in _cache:
                _cache[name] = abs_str
                count += 1
            else:
                logger.debug(f"Duplicate image filename '{name}', keeping first occurrence")
            if stem not in _stem_cache:
                _stem_cache[stem] = abs_str

    logger.info(f"Image cache built: {count} files from {base}")
    return _cache


def resolve(filename: str) -> str | None:
    """Return absolute path for a given image filename, or None if not found.
    Falls back to stem-only lookup for datasets that store filenames without extensions."""
    result = _cache.get(filename)
    if result is None:
        # Try treating filename as a stem (no extension)
        result = _stem_cache.get(filename)
    return result


def get_cache() -> dict[str, str]:
    return _cache
