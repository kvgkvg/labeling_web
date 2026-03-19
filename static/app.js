'use strict';

// ============================================================
// 1. CONSTANTS & STATE
// ============================================================

const HEARTBEAT_MS = 60_000;

const state = {
  userId:           null,
  currentSample:    null,
  currentImageIdx:  0,
  datasetFilter:    null,

  // Canvas
  mode:             'pan',    // 'pan' | 'draw'
  transform:        { scale: 1, offsetX: 0, offsetY: 0 },
  currentImg:       null,
  imgNaturalW:      0,
  imgNaturalH:      0,

  // Bboxes (normalized 0-1000)
  modelBboxes:      [],
  gtBboxes:         [],
  selectedGtIdx:    null,

  // Interaction
  isDragging:       false,
  dragStartCanvas:  null,
  dragStartOffset:  null,
  isDrawing:        false,
  drawStartNorm:    null,
  drawCurrentNorm:  null,

  // Bbox answer verdict (manual True/False, null = not set)
  bboxAnswerVerdict: null,

  // Reasoning originals
  originalVisualReasoning: '',
  originalReasoning:       '',

  // Timers
  heartbeatTimer: null,
};

// ============================================================
// 2. Text formatter for reasoning / visual reasoning
// ============================================================

function formatReasoningText(text) {
  if (!text) return text;
  let t = text.trim();

  // Normalize multiple spaces and strip leading/trailing spaces per sentence
  t = t.replace(/[ \t]+/g, ' ');

  // Insert blank line before top-level steps: "Step 1:", "Step 2.", etc.
  t = t.replace(/\s*(Step\s+\d+[.:]\s*)/gi, '\n\n$1');

  // Insert newline + bullet before sub-steps: "Sub-step 1.1:", "1.1.", "1.1:"
  t = t.replace(/\s*(Sub-step\s+[\d.]+[.:]\s*)/gi, '\n  • $1');

  // Insert newline + bullet before numbered items inside a step: "1. Text", "2. Text"
  // (only when preceded by space/period, avoid splitting decimals)
  t = t.replace(/(?<=[.!?])\s+(\d+\.\s+)/g, '\n  • $1');

  // Insert newline before "Note:", "Conclusion:", "Therefore:", "Answer:", "Final:"
  t = t.replace(/\s*((?:Note|Conclusion|Therefore|Answer|Final answer|Result)[.:]\s*)/gi, '\n\n→ $1');

  // Collapse 3+ newlines to 2
  t = t.replace(/\n{3,}/g, '\n\n');

  return t.trim();
}

function autoResizeTa(ta) {
  ta.style.height = 'auto';
  ta.style.height = ta.scrollHeight + 'px';
}

// ============================================================
// 3. IoU  (mirrors core/iou.py)
// ============================================================

function boxArea(b) {
  return Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
}

function singleIou(a, b) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  if (inter === 0) return 0;
  const union = boxArea(a) + boxArea(b) - inter;
  return union === 0 ? 0 : inter / union;
}

function computeIou(predicted, groundtruth) {
  if (!predicted.length && !groundtruth.length) return null;
  if (!predicted.length || !groundtruth.length) return 0;
  const sorted = [...predicted].sort((a, b) => boxArea(b) - boxArea(a));
  const unmatched = groundtruth.map((_, i) => i);
  let total = 0;
  for (const p of sorted) {
    if (!unmatched.length) break;
    let best = -1, bestIdx = -1;
    for (const gi of unmatched) {
      const v = singleIou(p, groundtruth[gi]);
      if (v > best) { best = v; bestIdx = gi; }
    }
    if (bestIdx >= 0) {
      total += best;
      unmatched.splice(unmatched.indexOf(bestIdx), 1);
    }
  }
  return total / Math.max(predicted.length, groundtruth.length);
}

// ============================================================
// 3. API HELPERS
// ============================================================

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function apiLoadNext() {
  const p = new URLSearchParams({ user_id: state.userId });
  if (state.datasetFilter) p.set('dataset', state.datasetFilter);
  const data = await apiFetch(`/api/samples/next?${p}`);
  return data.sample;
}

async function apiSave(sampleId, payload) {
  return apiFetch(`/api/labels/${sampleId}`, {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

async function apiUnlock(sampleId) {
  return apiFetch(`/api/samples/${sampleId}/unlock`, {
    method: 'POST',
    body: JSON.stringify({ user_id: state.userId }),
  });
}

async function apiHeartbeat(sampleId) {
  return apiFetch(`/api/samples/${sampleId}/heartbeat`, {
    method: 'POST',
    body: JSON.stringify({ user_id: state.userId }),
  }).catch(() => {});
}

async function apiProgress() {
  return apiFetch('/api/progress');
}

// ============================================================
// 4. CANVAS UTILITIES
// ============================================================

function getCanvas() { return document.getElementById('viewer-canvas'); }
function getCtx()    { return getCanvas().getContext('2d'); }

function canvasPos(canvas, evt) {
  const r = canvas.getBoundingClientRect();
  return { x: evt.clientX - r.left, y: evt.clientY - r.top };
}

function normToCanvas(nx, ny) {
  const t = state.transform;
  return {
    x: t.offsetX + (nx / 1000) * state.imgNaturalW * t.scale,
    y: t.offsetY + (ny / 1000) * state.imgNaturalH * t.scale,
  };
}

function canvasToNorm(cx, cy) {
  const t = state.transform;
  if (!state.imgNaturalW || !state.imgNaturalH) return { x: 0, y: 0 };
  return {
    x: Math.max(0, Math.min(1000, ((cx - t.offsetX) / (state.imgNaturalW  * t.scale)) * 1000)),
    y: Math.max(0, Math.min(1000, ((cy - t.offsetY) / (state.imgNaturalH * t.scale)) * 1000)),
  };
}

function fitImage() {
  const canvas = getCanvas();
  if (!state.currentImg || !canvas.width || !canvas.height) return;
  const { imgNaturalW: iw, imgNaturalH: ih } = state;
  if (!iw || !ih) return;
  const scale = Math.min(canvas.width / iw, canvas.height / ih) * 0.95;
  state.transform = {
    scale,
    offsetX: (canvas.width  - iw * scale) / 2,
    offsetY: (canvas.height - ih * scale) / 2,
  };
  render();
}

function zoomAt(cx, cy, factor) {
  const t = state.transform;
  const ns = Math.max(0.05, Math.min(20, t.scale * factor));
  const r  = ns / t.scale;
  state.transform = { scale: ns, offsetX: cx - r * (cx - t.offsetX), offsetY: cy - r * (cy - t.offsetY) };
  render();
}

// ============================================================
// 5. CANVAS RENDERING
// ============================================================

function render() {
  const canvas = getCanvas();
  const ctx    = getCtx();
  const { width: cw, height: ch } = canvas;

  ctx.clearRect(0, 0, cw, ch);
  ctx.fillStyle = '#13131f';
  ctx.fillRect(0, 0, cw, ch);

  if (!state.currentImg) {
    ctx.fillStyle = '#444466';
    ctx.font = '16px system-ui';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Load a sample to begin', cw / 2, ch / 2);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    return;
  }

  const t  = state.transform;
  const iw = state.imgNaturalW * t.scale;
  const ih = state.imgNaturalH * t.scale;

  ctx.drawImage(state.currentImg, t.offsetX, t.offsetY, iw, ih);

  // Model bboxes — blue dashed
  ctx.save();
  ctx.strokeStyle = '#89b4fa';
  ctx.lineWidth   = 2;
  ctx.setLineDash([6, 4]);
  for (const b of state.modelBboxes) {
    drawBoxRect(ctx, b, false, null);
  }
  ctx.restore();

  // GT bboxes — green solid
  ctx.save();
  ctx.setLineDash([]);
  for (let i = 0; i < state.gtBboxes.length; i++) {
    const sel = i === state.selectedGtIdx;
    ctx.strokeStyle = sel ? '#c0ffc0' : '#a6e3a1';
    ctx.lineWidth   = sel ? 3 : 2;
    drawBoxRect(ctx, state.gtBboxes[i], true, i);
  }
  ctx.restore();

  // Drawing preview
  if (state.isDrawing && state.drawStartNorm && state.drawCurrentNorm) {
    const a  = normToCanvas(state.drawStartNorm.x,   state.drawStartNorm.y);
    const b  = normToCanvas(state.drawCurrentNorm.x, state.drawCurrentNorm.y);
    const rx = Math.min(a.x, b.x), ry = Math.min(a.y, b.y);
    const rw = Math.abs(b.x - a.x), rh = Math.abs(b.y - a.y);
    ctx.save();
    ctx.strokeStyle = '#a6e3a1';
    ctx.lineWidth   = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();
  }
}

function drawBoxRect(ctx, bbox, showDelete, idx) {
  const tl = normToCanvas(bbox.x1, bbox.y1);
  const br = normToCanvas(bbox.x2, bbox.y2);
  const x  = Math.min(tl.x, br.x), y = Math.min(tl.y, br.y);
  const w  = Math.abs(br.x - tl.x), h = Math.abs(br.y - tl.y);
  ctx.strokeRect(x, y, w, h);

  // Label text
  if (bbox.label) {
    ctx.save();
    ctx.fillStyle    = ctx.strokeStyle;
    ctx.font         = '11px monospace';
    ctx.textBaseline = 'bottom';
    ctx.fillText(bbox.label, x + 3, y - 2);
    ctx.restore();
  }

  // Delete handle (×) at top-right corner
  if (showDelete && idx !== null) {
    const hx = x + w, hy = y;
    ctx.save();
    ctx.fillStyle = '#f38ba8';
    ctx.beginPath();
    ctx.arc(hx, hy, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle    = '#1e1e2e';
    ctx.font         = 'bold 11px sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('×', hx, hy);
    ctx.restore();
  }
}

// ============================================================
// 6. CANVAS EVENTS
// ============================================================

function initCanvasEvents() {
  const canvas = getCanvas();

  // Zoom on scroll
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const pos    = canvasPos(canvas, e);
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    zoomAt(pos.x, pos.y, factor);
  }, { passive: false });

  canvas.addEventListener('mousedown', (e) => {
    const pos = canvasPos(canvas, e);

    if (state.mode === 'pan') {
      state.isDragging      = true;
      state.dragStartCanvas = pos;
      state.dragStartOffset = { x: state.transform.offsetX, y: state.transform.offsetY };
      canvas.style.cursor   = 'grabbing';
      return;
    }

    // Draw mode — check delete handle first
    const delIdx = getDeleteHandleAt(pos.x, pos.y);
    if (delIdx !== null) {
      state.gtBboxes.splice(delIdx, 1);
      if (state.selectedGtIdx === delIdx) state.selectedGtIdx = null;
      else if (state.selectedGtIdx > delIdx) state.selectedGtIdx--;
      render();
      updateBboxBadges();
      return;
    }

    if (!state.currentImg) return;
    state.isDrawing     = true;
    const norm          = canvasToNorm(pos.x, pos.y);
    state.drawStartNorm = norm;
    state.drawCurrentNorm = norm;
  });

  canvas.addEventListener('mousemove', (e) => {
    const pos = canvasPos(canvas, e);
    if (state.isDragging) {
      state.transform.offsetX = state.dragStartOffset.x + (pos.x - state.dragStartCanvas.x);
      state.transform.offsetY = state.dragStartOffset.y + (pos.y - state.dragStartCanvas.y);
      render();
    } else if (state.isDrawing) {
      state.drawCurrentNorm = canvasToNorm(pos.x, pos.y);
      render();
    }
  });

  canvas.addEventListener('mouseup', (e) => {
    if (state.isDragging) {
      state.isDragging    = false;
      canvas.style.cursor = 'grab';
      return;
    }
    if (state.isDrawing) {
      state.isDrawing = false;
      const s = state.drawStartNorm, c = state.drawCurrentNorm;
      if (s && c) {
        const x1 = Math.min(s.x, c.x), y1 = Math.min(s.y, c.y);
        const x2 = Math.max(s.x, c.x), y2 = Math.max(s.y, c.y);
        // Require a minimum box size (5 units in normalized space)
        if ((x2 - x1) > 5 && (y2 - y1) > 5) {
          state.gtBboxes.push({ x1, y1, x2, y2, label: '' });
          updateBboxBadges();
        }
      }
      state.drawStartNorm   = null;
      state.drawCurrentNorm = null;
      render();
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (state.isDragging) {
      state.isDragging    = false;
      canvas.style.cursor = state.mode === 'pan' ? 'grab' : 'crosshair';
    }
    if (state.isDrawing) {
      state.isDrawing       = false;
      state.drawStartNorm   = null;
      state.drawCurrentNorm = null;
      render();
    }
  });

  // Click to select GT bbox in draw mode
  canvas.addEventListener('click', (e) => {
    if (state.mode !== 'draw') return;
    const pos    = canvasPos(canvas, e);
    if (getDeleteHandleAt(pos.x, pos.y) !== null) return;
    const norm   = canvasToNorm(pos.x, pos.y);
    for (let i = state.gtBboxes.length - 1; i >= 0; i--) {
      const b = state.gtBboxes[i];
      if (norm.x >= b.x1 && norm.x <= b.x2 && norm.y >= b.y1 && norm.y <= b.y2) {
        state.selectedGtIdx = i;
        render();
        return;
      }
    }
    state.selectedGtIdx = null;
    render();
  });
}

function getDeleteHandleAt(cx, cy) {
  for (let i = state.gtBboxes.length - 1; i >= 0; i--) {
    const b  = state.gtBboxes[i];
    const tr = normToCanvas(b.x2, b.y1);      // top-right corner
    if (Math.hypot(cx - tr.x, cy - tr.y) <= 10) return i;
  }
  return null;
}

// ============================================================
// 7. TOOLBAR
// ============================================================

function setMode(mode) {
  state.mode = mode;
  const canvas = getCanvas();
  canvas.style.cursor = mode === 'pan' ? 'grab' : 'crosshair';
  document.getElementById('tool-pan').classList.toggle('active',  mode === 'pan');
  document.getElementById('tool-draw').classList.toggle('active', mode === 'draw');
}

function initToolbar() {
  document.getElementById('tool-pan').addEventListener('click',  () => setMode('pan'));
  document.getElementById('tool-draw').addEventListener('click', () => setMode('draw'));

  document.getElementById('tool-delete').addEventListener('click', () => {
    if (state.selectedGtIdx !== null) {
      state.gtBboxes.splice(state.selectedGtIdx, 1);
      state.selectedGtIdx = null;
      render();
      updateBboxBadges();
    }
  });

  document.getElementById('tool-clear').addEventListener('click', () => {
    state.gtBboxes      = [];
    state.selectedGtIdx = null;
    render();
    updateBboxBadges();
  });

  document.getElementById('tool-zoom-in').addEventListener('click', () => {
    const c = getCanvas();
    zoomAt(c.width / 2, c.height / 2, 1.25);
  });
  document.getElementById('tool-zoom-out').addEventListener('click', () => {
    const c = getCanvas();
    zoomAt(c.width / 2, c.height / 2, 0.8);
  });
  document.getElementById('tool-fit').addEventListener('click', fitImage);
}

// ============================================================
// 8. KEYBOARD SHORTCUTS
// ============================================================

function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;

    switch (e.key) {
      case 'd': case 'D': setMode('draw');     break;
      case 'p': case 'P': setMode('pan');      break;
      case 'f': case 'F': fitImage();          break;
      case 'Escape':       setMode('pan');     break;
      case '+': case '=': {
        const c = getCanvas();
        zoomAt(c.width / 2, c.height / 2, 1.25);
        break;
      }
      case '-': {
        const c = getCanvas();
        zoomAt(c.width / 2, c.height / 2, 0.8);
        break;
      }
      case 'Delete': case 'Backspace': {
        if (state.selectedGtIdx !== null) {
          state.gtBboxes.splice(state.selectedGtIdx, 1);
          state.selectedGtIdx = null;
          render();
          updateBboxBadges();
        }
        break;
      }
    }
  });
}

// ============================================================
// 9. CANVAS RESIZE
// ============================================================

function initCanvasResize() {
  const container = document.getElementById('canvas-container');
  const canvas    = getCanvas();

  const resize = () => {
    const r = container.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return;
    canvas.width  = Math.floor(r.width);
    canvas.height = Math.floor(r.height);
    fitImage();
  };

  new ResizeObserver(resize).observe(container);
  resize();
}

// ============================================================
// 10. LIVE BADGES
// ============================================================

function setBadge(id, value, trueText = '✓', falseText = '✗') {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'badge';
  if (value === null || value === undefined) {
    el.textContent = '— N/A';
    el.classList.add('badge-unknown');
  } else if (value === true) {
    el.textContent = trueText;
    el.classList.add('badge-true');
  } else {
    el.textContent = falseText;
    el.classList.add('badge-false');
  }
}

function updateBboxBadges() {
  const iou       = computeIou(state.modelBboxes, state.gtBboxes);
  const threshold = 0.8;
  const correct   = iou === null ? null : iou > threshold;
  const iouLabel  = iou !== null ? ` (IoU ${iou.toFixed(2)})` : '';
  setBadge('badge-bbox-correct', correct,
    `✓ Correct${iouLabel}`,
    `✗ Wrong${iouLabel}`
  );

  // Reflect bbox answer verdict toggle
  document.getElementById('bbox-answer-true').classList.toggle('active-true',   state.bboxAnswerVerdict === true);
  document.getElementById('bbox-answer-false').classList.toggle('active-false', state.bboxAnswerVerdict === false);

  const cnt = state.gtBboxes.length;
  document.getElementById('bbox-count-display').textContent =
    `${cnt} GT bbox${cnt !== 1 ? 'es' : ''}`;
}

function updateReasoningBadges() {
  const vrVal = document.getElementById('visual-reasoning-text').value;
  const rVal  = document.getElementById('reasoning-text').value;
  const vrOk  = vrVal.trim() === state.originalVisualReasoning.trim();
  const rOk   = rVal.trim()  === state.originalReasoning.trim();
  setBadge('badge-visual-reasoning-correct', vrOk, '✓ Unchanged', '✗ Edited');
  setBadge('badge-reasoning-correct',        rOk,  '✓ Unchanged', '✗ Edited');
  document.getElementById('visual-reasoning-text').classList.toggle('dirty', !vrOk);
  document.getElementById('reasoning-text').classList.toggle('dirty', !rOk);
}

function updateStaticBadges(sample) {
  const mcq    = (sample.mcq_answer   || '').trim().toUpperCase();
  const bboxA  = (sample.bbox_answer  || '').trim().toUpperCase();
  const reasonA = (sample.reason_answer || '').trim().toUpperCase();
  const bboxChanged   = (bboxA && mcq)   ? bboxA   !== mcq : null;
  const reasonChanged = (reasonA && mcq) ? reasonA !== mcq : null;
  // badge-bbox-answer-changed: true = answers are SAME (consistent)
  setBadge('badge-bbox-answer-changed',   bboxChanged   === null ? null : !bboxChanged,   '✓ Consistent', '✗ Changed');
  setBadge('badge-reason-answer-changed', reasonChanged === null ? null : !reasonChanged, '✓ Consistent', '✗ Changed');
}

// ============================================================
// 11. UI POPULATION
// ============================================================

function populateSample(sample) {
  state.currentSample          = sample;
  state.gtBboxes               = [];
  state.modelBboxes            = sample.extracted_bbox || [];
  state.selectedGtIdx          = null;
  state.bboxAnswerVerdict      = null;
  state.originalVisualReasoning = formatReasoningText(sample.visual_reasoning || '');
  state.originalReasoning       = formatReasoningText(sample.reasoning        || '');
  state.currentImageIdx        = 0;

  // Header
  document.getElementById('sample-id').textContent   = sample.id;
  document.getElementById('dataset-name').textContent = sample.dataset;

  // Status banner
  const banner = document.getElementById('status-banner');
  if (!sample.is_valid) {
    banner.textContent = `⚠ Invalid sample — bbox: ${sample.bbox_status}, reasoning: ${sample.reasoning_status}, answer: ${sample.answer_status}`;
    banner.className   = 'status-banner warning';
  } else {
    banner.className = 'hidden';
  }

  // ── Part 1: MCQ ──────────────────────────────────────────
  document.getElementById('question-text').textContent = sample.question || '(no question)';

  const choicesList = document.getElementById('choices-list');
  choicesList.innerHTML = '';
  for (const [key, val] of Object.entries(sample.choices || {})) {
    const isGt    = key === (sample.answer    || '').trim();
    const isModel = key === (sample.mcq_answer || '').trim();
    const row     = document.createElement('div');
    row.className = 'choice-row';
    const keySpan = document.createElement('span');
    let cls = 'choice-key';
    if (isGt && isModel) cls += ' choice-gt choice-model';
    else if (isGt)       cls += ' choice-gt';
    else if (isModel)    cls += ' choice-model';
    keySpan.className   = cls;
    keySpan.textContent = key;
    const valSpan = document.createElement('span');
    valSpan.className   = 'choice-val';
    valSpan.textContent = val || '';
    row.appendChild(keySpan);
    row.appendChild(valSpan);
    choicesList.appendChild(row);
  }

  const mcqEl  = document.getElementById('mcq-answer');
  mcqEl.textContent = sample.mcq_answer || '—';
  const correct = sample.mcq_answer && sample.answer &&
                  sample.mcq_answer.trim().toUpperCase() === sample.answer.trim().toUpperCase();
  mcqEl.className = 'ans-badge ' + (correct ? 'ans-correct' : 'ans-wrong');
  document.getElementById('groundtruth-answer').textContent = sample.answer || '—';

  // ── Part 2: BBOX ─────────────────────────────────────────
  document.getElementById('bbox-answer-display').textContent = sample.bbox_answer || '—';
  const vrTa = document.getElementById('visual-reasoning-text');
  vrTa.value = state.originalVisualReasoning;
  vrTa.classList.remove('dirty');
  autoResizeTa(vrTa);

  // ── Part 3: Reasoning ────────────────────────────────────
  document.getElementById('reason-answer-display').textContent = sample.reason_answer || '—';
  const rTa = document.getElementById('reasoning-text');
  rTa.value = state.originalReasoning;
  rTa.classList.remove('dirty');
  autoResizeTa(rTa);

  // Badges
  updateStaticBadges(sample);
  updateBboxBadges();
  updateReasoningBadges();

  // Image tabs + load first image
  buildImageTabs(sample.images || []);
  loadImage(0);

  // Enable actions
  document.getElementById('btn-save').disabled = false;
  document.getElementById('btn-skip').disabled = false;
}

function buildImageTabs(images) {
  const tabsEl = document.getElementById('image-tabs');
  tabsEl.innerHTML = '';
  if (images.length <= 1) {
    tabsEl.style.display = 'none';
    return;
  }
  tabsEl.style.display = 'flex';
  images.forEach((filename, i) => {
    const tab = document.createElement('button');
    tab.className   = 'image-tab' + (i === 0 ? ' active' : '');
    tab.textContent = `Image ${i + 1}`;
    tab.title       = filename;
    tab.addEventListener('click', () => {
      state.currentImageIdx = i;
      document.querySelectorAll('.image-tab').forEach((t, j) => t.classList.toggle('active', j === i));
      loadImage(i);
    });
    tabsEl.appendChild(tab);
  });
}

function loadImage(idx) {
  const images = (state.currentSample && state.currentSample.images) || [];
  if (!images.length) {
    state.currentImg  = null;
    state.imgNaturalW = 0;
    state.imgNaturalH = 0;
    render();
    return;
  }
  const filename = images[idx] || images[0];
  const img      = new Image();
  img.onload = () => {
    state.currentImg  = img;
    state.imgNaturalW = img.naturalWidth;
    state.imgNaturalH = img.naturalHeight;
    fitImage();
  };
  img.onerror = () => {
    state.currentImg  = null;
    state.imgNaturalW = 0;
    state.imgNaturalH = 0;
    render();
    showToast(`Image not found: ${filename}`, 'warning');
  };
  img.src = `/images/${encodeURIComponent(filename)}`;
}

// ============================================================
// 12. SAMPLE LIFECYCLE
// ============================================================

async function loadNext() {
  setActionsEnabled(false);
  try {
    const sample = await apiLoadNext();
    if (!sample) {
      showEmpty();
      return;
    }
    populateSample(sample);
    startHeartbeat(sample.id);
    await refreshProgress();
  } catch (err) {
    showToast(`Failed to load sample: ${err.message}`, 'error');
  } finally {
    // actions are re-enabled inside populateSample on success
  }
}

async function saveAndNext() {
  if (!state.currentSample) return;
  setActionsEnabled(false);

  const payload = {
    user_id:                 state.userId,
    groundtruth_bboxes:      state.gtBboxes,
    bbox_answer_verdict:     state.bboxAnswerVerdict,
    edited_visual_reasoning: document.getElementById('visual-reasoning-text').value,
    edited_reasoning:        document.getElementById('reasoning-text').value,
    groundtruth_answer:      null,
  };

  try {
    await apiSave(state.currentSample.id, payload);
    showToast('Label saved!', 'success');
    stopHeartbeat();
    clearViewer();
    await loadNext();
  } catch (err) {
    showToast(`Save failed: ${err.message}`, 'error');
    setActionsEnabled(true);
  }
}

async function skipSample() {
  if (!state.currentSample) return;
  const id = state.currentSample.id;
  setActionsEnabled(false);
  stopHeartbeat();
  clearViewer();
  try { await apiUnlock(id); } catch (_) {}
  await loadNext();
}

function clearViewer() {
  state.currentSample  = null;
  state.currentImg     = null;
  state.imgNaturalW    = 0;
  state.imgNaturalH    = 0;
  state.gtBboxes       = [];
  state.modelBboxes    = [];
  state.selectedGtIdx  = null;
  document.getElementById('sample-id').textContent   = '—';
  document.getElementById('dataset-name').textContent = '—';
  render();
}

function showEmpty() {
  const banner = document.getElementById('status-banner');
  banner.textContent = 'No available samples — all labeled or currently locked. Try again shortly.';
  banner.className   = 'status-banner info';
  document.getElementById('sample-id').textContent   = 'Done';
  document.getElementById('dataset-name').textContent = '—';
}

function setActionsEnabled(enabled) {
  document.getElementById('btn-save').disabled = !enabled;
  document.getElementById('btn-skip').disabled = !enabled;
}

// ============================================================
// 13. HEARTBEAT
// ============================================================

function startHeartbeat(sampleId) {
  stopHeartbeat();
  state.heartbeatTimer = setInterval(() => apiHeartbeat(sampleId), HEARTBEAT_MS);
}

function stopHeartbeat() {
  if (state.heartbeatTimer) {
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = null;
  }
}

// ============================================================
// 14. PROGRESS
// ============================================================

async function refreshProgress() {
  try {
    const data         = await apiProgress();
    let totalLabeled   = 0, totalValid = 0;
    for (const v of Object.values(data)) {
      totalLabeled += v.labeled;
      totalValid   += v.total;
    }
    const pct = totalValid > 0 ? (totalLabeled / totalValid * 100) : 0;
    document.getElementById('progress-text').textContent =
      `${totalLabeled} / ${totalValid} (${pct.toFixed(0)}%)`;
    document.getElementById('progress-bar').style.width = `${pct}%`;
  } catch (_) {}
}

// ============================================================
// 15. DATASET FILTER
// ============================================================

async function initDatasetFilter() {
  try {
    const data   = await apiProgress();
    const select = document.getElementById('dataset-filter');
    select.innerHTML = '<option value="">All datasets</option>';
    for (const [ds, v] of Object.entries(data)) {
      const opt = document.createElement('option');
      opt.value       = ds;
      opt.textContent = `${ds.replace('GeminiFlashLite2-5_', '')} (${v.labeled}/${v.total})`;
      select.appendChild(opt);
    }
    select.addEventListener('change', () => {
      state.datasetFilter = select.value || null;
    });
  } catch (_) {}
}

// ============================================================
// 16. TOAST NOTIFICATIONS
// ============================================================

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast     = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('toast-visible')));
  setTimeout(() => {
    toast.classList.remove('toast-visible');
    setTimeout(() => toast.remove(), 350);
  }, 3000);
}

// ============================================================
// 17. USER ID
// ============================================================

function initUserId() {
  const stored = localStorage.getItem('vqa_user_id');
  if (!stored) {
    document.getElementById('user-modal').style.display = 'flex';

    const submit = () => {
      const val = document.getElementById('user-id-input').value.trim();
      if (!val) return;
      localStorage.setItem('vqa_user_id', val);
      state.userId = val;
      document.getElementById('user-id-display').textContent = val;
      document.getElementById('user-modal').style.display    = 'none';
      onReady();
    };

    document.getElementById('user-id-submit').addEventListener('click', submit);
    document.getElementById('user-id-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submit();
    });
  } else {
    state.userId = stored;
    document.getElementById('user-id-display').textContent  = stored;
    document.getElementById('user-modal').style.display = 'none';
    onReady();
  }
}

async function onReady() {
  await initDatasetFilter();
  await refreshProgress();
  await loadNext();
}

// ============================================================
// 18. BEFORE UNLOAD
// ============================================================

function initBeforeUnload() {
  window.addEventListener('beforeunload', () => {
    if (!state.currentSample || !state.userId) return;
    const body = JSON.stringify({ user_id: state.userId });
    navigator.sendBeacon(
      `/api/samples/${state.currentSample.id}/unlock`,
      new Blob([body], { type: 'application/json' })
    );
  });
}

// ============================================================
// 19. INIT
// ============================================================

function init() {
  initCanvasResize();
  initCanvasEvents();
  initToolbar();
  initKeyboard();
  initBeforeUnload();

  // Reasoning textarea listeners
  const vrTaEl = document.getElementById('visual-reasoning-text');
  const rTaEl  = document.getElementById('reasoning-text');
  vrTaEl.addEventListener('input', () => { autoResizeTa(vrTaEl); updateReasoningBadges(); });
  rTaEl .addEventListener('input', () => { autoResizeTa(rTaEl);  updateReasoningBadges(); });

  // Reset buttons
  document.getElementById('btn-reset-visual-reasoning').addEventListener('click', () => {
    vrTaEl.value = state.originalVisualReasoning;
    autoResizeTa(vrTaEl);
    updateReasoningBadges();
  });
  document.getElementById('btn-reset-reasoning').addEventListener('click', () => {
    rTaEl.value = state.originalReasoning;
    autoResizeTa(rTaEl);
    updateReasoningBadges();
  });

  // Bbox answer verdict toggle
  document.getElementById('bbox-answer-true').addEventListener('click', () => {
    state.bboxAnswerVerdict = state.bboxAnswerVerdict === true ? null : true;
    updateBboxBadges();
  });
  document.getElementById('bbox-answer-false').addEventListener('click', () => {
    state.bboxAnswerVerdict = state.bboxAnswerVerdict === false ? null : false;
    updateBboxBadges();
  });

  // Action buttons
  document.getElementById('btn-save').addEventListener('click', saveAndNext);
  document.getElementById('btn-skip').addEventListener('click', skipSample);
  document.getElementById('btn-next-sample').addEventListener('click', () => {
    if (state.currentSample) skipSample();
    else loadNext();
  });

  // Initial blank render
  render();

  // Start user flow
  initUserId();
}

document.addEventListener('DOMContentLoaded', init);
