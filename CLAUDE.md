# CLAUDE.md — VQA Label Website

## Quick Start
```bash
conda activate label_website   # Python 3.11
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

---

## Build Status

| Stage | Status |
|-------|--------|
| 1 — Core Data Pipeline | ✅ COMPLETE |
| 2 — Backend Services | ✅ COMPLETE |
| 3 — Frontend UI | ✅ COMPLETE |
| 4 — Test Tooling (`scripts/generate_sample_data.py`) | ✅ COMPLETE |

**Verified data:** 15,793 total samples, 14,584 valid.

---

## Tech Stack
- **Backend**: Python + FastAPI + uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (no framework)
- **Storage**: XLSX input (read-only); labels as JSON files; no DB

---

## Project Structure
```
label_website/
├── .env                  # IMAGE_BASE_DIR=~/LMUData, IOU_THRESHOLD=0.8, LOCK_TIMEOUT_SECONDS=300
├── main.py               # FastAPI entry; startup wires image_resolver+data_loader+lock_manager
├── core/
│   ├── data_loader.py    # Load+merge XLSX; exposes load_all(), get_sample(), list_samples()
│   ├── bbox_parser.py    # 4 bbox string formats → [{x1,y1,x2,y2,label}]
│   ├── iou.py            # Greedy IoU, normalized 0-1000 coords
│   ├── image_resolver.py # Recursive scan IMAGE_BASE_DIR → {basename: abs_path} + stem fallback
│   ├── lock_manager.py   # File-based locking; single labels/locks.json; 300s heartbeat timeout
│   └── auto_label.py     # bbox_correct (IoU), answer change detection, reasoning diff
├── routers/
│   ├── samples.py        # list/get/next/lock/unlock/heartbeat/lock-status/progress
│   └── labels.py         # save/load per-dataset JSON with fcntl.flock
├── data/
│   ├── bbox/             # 6 raw XLSX + bbox_filter.xlsx (6 sheets)
│   ├── reasoning/        # 6 raw XLSX + reasoning_filter.xlsx (6 sheets)
│   └── answer/           # 6 raw XLSX + answer_filter.xlsx (6 sheets)
├── labels/               # {dataset}.json (output) + locks.json
└── static/               # index.html, app.js, style.css
```

---

## Data & Merge

**6 datasets** (all prefixed `GeminiFlashLite2-5_`): `BLINK`, `MMBench_DEV_EN_V11`, `MME-RealWorld`, `MME`, `MMStar`, `SEEDBench_IMG`

**Merge key**: `(dataset_name, row_idx)` — dataset_name = raw filename stem = filter sheet name; row_idx = 1-based Row column in filter files.

**Filter column renames** (critical — both bbox and reasoning filters have `extracted_reasoning` with different meanings):
- `bbox_filter.extracted_reasoning` → `visual_reasoning`
- `reasoning_filter.extracted_reasoning` → `reasoning`
- `bbox_filter.extracted_answer` → `bbox_answer`
- `reasoning_filter.extracted_answer` → `reason_answer`
- `answer_filter.extracted_answer` → `mcq_answer`

**`is_valid`**: all three statuses start with `"Valid"` (values like `"Valid (single_letter)"` are valid).

**DATASET_SCHEMA** (`core/data_loader.py` lines 30–73):
| Dataset | choices | image_path col |
|---------|---------|----------------|
| BLINK | A,B,C,D | `image_path` (ast.literal_eval needed — single-quote list repr) |
| MMBench_DEV_EN_V11 | A,B,C,D | None (use index value) |
| MME-RealWorld | A,B,C,D,E | `image_path` (absolute path string) |
| MME | [] (yes/no) | `image_path` |
| MMStar | A,B,C,D | None (use index value) |
| SEEDBench_IMG | A,B,C,D | None (use index value) |

**Image resolver quirk**: datasets MMBench_V11, MMStar, SEEDBench_IMG store filenames without extensions → `image_resolver` builds `_stem_cache` (stem→abs_path) as fallback.

**Actual IMAGE_BASE_DIR**: `~/LMUData` (not `~/LMUData/images`) — images at `~/LMUData/{DatasetName}/`.

---

## API
```
GET  /api/samples                        # list all with status
GET  /api/samples/next?user_id=&dataset= # next unlocked+unlabeled for user
GET  /api/samples/{id}                   # full sample
POST /api/samples/{id}/lock              # acquire {user_id}
POST /api/samples/{id}/unlock            # release {user_id}
POST /api/samples/{id}/heartbeat         # refresh lock
GET  /api/samples/{id}/lock-status
GET  /api/progress                       # per-dataset labeled/total
GET  /api/labels/{id}
POST /api/labels/{id}                    # save label (validates lock ownership)
POST /api/admin/reload                   # hot-reload XLSX without restart
GET  /images/{filename}                  # serve image via resolver cache
```

---

## Label Output Format (`labels/{dataset}.json`)
```json
{
  "1": {
    "id": "GeminiFlashLite2-5_BLINK_1", "dataset": "...", "row_idx": 1,
    "labeled_by": "alice", "labeled_at": "...",
    "bbox_labels": {
      "model_bboxes": [...], "groundtruth_bboxes": [...], "iou_score": 0.94, "bbox_correct": true
    },
    "bbox_reasoning_labels": { "original_visual_reasoning": "...", "edited_visual_reasoning": "...", "visual_reasoning_correct": true },
    "reasoning_labels": { "original_reasoning": "...", "edited_reasoning": "...", "reasoning_correct": true },
    "mcq_labels": { "groundtruth": "B", "mcq_answer": "A" },
    "bbox_answer_verdict": true    // manual True/False/null set by labeler
  }
}
```

---

## Frontend (static/)

**Layout**: Left panel 60% (MCQ + Bbox + Reasoning cards) | Right panel 40% (canvas image viewer)

**3 annotation parts:**
1. **MCQ** — question, choices (A–E), mcq_answer vs groundtruth badges
2. **Bbox** — IoU auto-badge for bbox_correct; manual ✓/✗ icon buttons for `bbox_answer_verdict`; visual_reasoning textarea (auto-formatted + auto-resize)
3. **Reasoning** — reasoning textarea (auto-formatted + auto-resize); reset button

**Canvas**: zoom/pan/draw GT bboxes; model bboxes in blue dashed, GT in green solid; coords in 0–1000 normalized space; multi-image tabs.

**Text formatting** (`formatReasoningText()` in app.js): inserts blank lines before `Step N:`, bullets before `Sub-step N.N:`, `→` before conclusion keywords. Textareas auto-resize to content (no fixed rows).

**Multi-user**: user_id in localStorage; heartbeat every 60s; sendBeacon unlock on beforeunload.

---

## Key Quirks & Gotchas

| Quirk | Detail |
|-------|--------|
| `answer_status` validity | Use `startswith("Valid")` not `== "Valid"` |
| BLINK image_path | Single-quote list repr → needs `ast.literal_eval` |
| Extension-less filenames | `_stem_cache` fallback in image_resolver |
| Filter sheet name length | Excel caps at 31 chars; match via `full_stem[:31]` |
| Both filter files have `extracted_reasoning` | Rename immediately on read — different fields |
| Bbox coords | Normalized 0–1000, not pixels |
| `.env` tilde | Must call `os.path.expanduser()` — not auto-expanded |
