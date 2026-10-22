/**
 * AI CCTV Platform — Zone Editor (Phase 4a)
 * Interactive polygon vertex editing on a paused video frame.
 * Works on a draft copy; changes are persisted only on Save via PUT /api/zones.
 */

class ZoneEditor {
  constructor() {
    // DOM refs — set when view opens
    this.container = null;     // #zone-editor-view
    this.canvas = null;        // #ze-canvas
    this.ctx = null;
    this.video = null;         // shared <video> element
    this.frameSlider = null;   // #ze-frame-slider
    this.zoneListEl = null;    // #ze-zone-list
    this.propsEl = null;       // #ze-props-panel
    this.toastEl = null;       // #ze-toast

    // State
    this.draft = [];           // Array of zone draft objects (deep copy from store)
    this.savedSnapshot = '';   // JSON string of last-saved state for dirty detection
    this.selectedZoneId = null;
    this.hoveredHandle = null; // { zoneIdx, vertexIdx }
    this.dragging = null;      // { zoneIdx, vertexIdx, startNorm: [x,y] }
    this.isOpen = false;
    this.suppressBeforeUnload = false;

    // Bind methods for event listeners
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onKeyDown = this._onKeyDown.bind(this);
    this._onResize = this._onResize.bind(this);
    this._onBeforeUnload = this._onBeforeUnload.bind(this);
  }

  // ────────────────────────────────────────────
  // Open / Close
  // ────────────────────────────────────────────

  open() {
    this.container = document.getElementById('zone-editor-view');
    this.canvas = document.getElementById('ze-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.frameSlider = document.getElementById('ze-frame-slider');
    this.zoneListEl = document.getElementById('ze-zone-list');
    this.propsEl = document.getElementById('ze-props-panel');
    this.toastEl = document.getElementById('ze-toast');

    // Use the main video element (already has a source loaded)
    this.video = document.getElementById('main-video');

    if (!this.video || !this.video.src || this.video.readyState < 1) {
      this._showToast('Please select a video first.', 'error');
      return false;
    }

    // Pause playback
    this.video.pause();
    if (window.cctvPlayer && window.cctvPlayer.isLoopRunning) {
      window.cctvPlayer.stopInferenceLoop && window.cctvPlayer.stopInferenceLoop();
    }

    // Load draft from zoneStore
    this._loadDraft();

    // Configure frame slider
    const dur = this.video.duration || 1;
    this.frameSlider.max = dur;
    this.frameSlider.step = 0.04; // ~1 frame at 25fps
    this.frameSlider.value = this.video.currentTime;
    this.frameSlider.addEventListener('input', () => {
      this.video.currentTime = parseFloat(this.frameSlider.value);
    });
    // Re-render when frame settles
    this.video.addEventListener('seeked', this._onSeeked = () => this.render());

    // Canvas events
    this.canvas.addEventListener('pointerdown', this._onPointerDown);
    this.canvas.addEventListener('pointermove', this._onPointerMove);
    this.canvas.addEventListener('pointerup', this._onPointerUp);
    document.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('resize', this._onResize);
    window.addEventListener('beforeunload', this._onBeforeUnload);

    this.isOpen = true;
    this.selectedZoneId = null;
    this._resizeCanvas();
    this._buildZoneList();
    this._buildPropsPanel();
    this.render();
    return true;
  }

  close() {
    if (!this.isOpen) return;
    // Remove event listeners
    if (this.canvas) {
      this.canvas.removeEventListener('pointerdown', this._onPointerDown);
      this.canvas.removeEventListener('pointermove', this._onPointerMove);
      this.canvas.removeEventListener('pointerup', this._onPointerUp);
    }
    if (this._onSeeked) {
      this.video.removeEventListener('seeked', this._onSeeked);
    }
    document.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('beforeunload', this._onBeforeUnload);
    this.isOpen = false;
    this.dragging = null;
    this.hoveredHandle = null;
  }

  isDirty() {
    return JSON.stringify(this.draft) !== this.savedSnapshot;
  }

  confirmDiscard() {
    if (!this.isDirty()) return true;
    return confirm('You have unsaved zone changes. Discard them?');
  }

  // ────────────────────────────────────────────
  // Draft management
  // ────────────────────────────────────────────

  _loadDraft() {
    const zones = window.zoneStore ? window.zoneStore.getZones() : [];
    this.draft = zones.map(z => ({
      id: z.id,
      name: z.name,
      color: z.color,
      polygon: z.polygon.map(pt => [...pt]),
      assignment_mode: z.assignment_mode || 'center',
      enabled: z.enabled,
    }));
    this.savedSnapshot = JSON.stringify(this.draft);
  }

  _getDraftById(id) {
    return this.draft.find(z => z.id === id) || null;
  }

  // ────────────────────────────────────────────
  // Canvas helpers (reuse getVideoRenderRect logic)
  // ────────────────────────────────────────────

  _resizeCanvas() {
    if (!this.canvas || !this.video) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
  }

  _getVideoRect() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const vw = this.video.videoWidth || 640;
    const vh = this.video.videoHeight || 480;
    const videoRatio = vw / vh;
    const containerRatio = cw / ch;

    let renderW, renderH, offsetX, offsetY;
    if (containerRatio > videoRatio) {
      renderH = ch;
      renderW = ch * videoRatio;
      offsetX = (cw - renderW) / 2;
      offsetY = 0;
    } else {
      renderW = cw;
      renderH = cw / videoRatio;
      offsetX = 0;
      offsetY = (ch - renderH) / 2;
    }
    return { renderW, renderH, offsetX, offsetY, vw, vh };
  }

  _normToCanvas(nx, ny) {
    const { renderW, renderH, offsetX, offsetY } = this._getVideoRect();
    return { x: offsetX + nx * renderW, y: offsetY + ny * renderH };
  }

  _canvasToNorm(cx, cy) {
    const { renderW, renderH, offsetX, offsetY } = this._getVideoRect();
    let nx = (cx - offsetX) / renderW;
    let ny = (cy - offsetY) / renderH;
    // Clamp to 0-1
    nx = Math.max(0, Math.min(1, nx));
    ny = Math.max(0, Math.min(1, ny));
    return [nx, ny];
  }

  // ────────────────────────────────────────────
  // Render
  // ────────────────────────────────────────────

  render() {
    if (!this.ctx || !this.canvas) return;
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    this.ctx.clearRect(0, 0, cw, ch);

    // Draw paused video frame onto canvas
    const { renderW, renderH, offsetX, offsetY } = this._getVideoRect();
    if (this.video.readyState >= 2) {
      this.ctx.drawImage(this.video, offsetX, offsetY, renderW, renderH);
    } else {
      this.ctx.fillStyle = '#111';
      this.ctx.fillRect(offsetX, offsetY, renderW, renderH);
    }

    // Draw black bars
    this.ctx.fillStyle = '#000';
    if (offsetX > 0) {
      this.ctx.fillRect(0, 0, offsetX, ch);
      this.ctx.fillRect(offsetX + renderW, 0, cw - offsetX - renderW, ch);
    }
    if (offsetY > 0) {
      this.ctx.fillRect(0, 0, cw, offsetY);
      this.ctx.fillRect(0, offsetY + renderH, cw, ch - offsetY - renderH);
    }

    const labelFontSize = Math.max(11, Math.min(14, Math.round(renderW / 55)));
    const labelFont = `bold ${labelFontSize}px JetBrains Mono, monospace`;

    // Draw each zone
    this.draft.forEach((zone, zIdx) => {
      const poly = zone.polygon;
      if (!poly || poly.length < 3) return;

      const isSelected = zone.id === this.selectedZoneId;
      const selfIntersecting = this._polygonSelfIntersects(poly);
      const canvasPts = poly.map(pt => this._normToCanvas(pt[0], pt[1]));

      // Fill
      this.ctx.beginPath();
      this.ctx.moveTo(canvasPts[0].x, canvasPts[0].y);
      for (let i = 1; i < canvasPts.length; i++) {
        this.ctx.lineTo(canvasPts[i].x, canvasPts[i].y);
      }
      this.ctx.closePath();

      if (!zone.enabled) {
        this.ctx.fillStyle = 'rgba(128, 128, 128, 0.08)';
        this.ctx.fill();
        this.ctx.setLineDash([6, 4]);
        this.ctx.strokeStyle = '#555';
        this.ctx.lineWidth = 1;
        this.ctx.stroke();
        this.ctx.setLineDash([]);
      } else {
        const fillAlpha = isSelected ? 0.22 : 0.12;
        this.ctx.fillStyle = this._hexToRGBA(zone.color, fillAlpha);
        this.ctx.fill();

        if (selfIntersecting) {
          this.ctx.strokeStyle = '#ff2222';
          this.ctx.lineWidth = 3;
        } else {
          this.ctx.strokeStyle = zone.color;
          this.ctx.lineWidth = isSelected ? 3 : 1.8;
        }
        this.ctx.stroke();
      }

      // Label (inside bounding box top-left)
      let minX = Infinity, minY = Infinity, maxX = -Infinity;
      for (const pt of canvasPts) {
        if (pt.x < minX) minX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.x > maxX) maxX = pt.x;
      }
      this.ctx.font = labelFont;
      let labelText = zone.name.toUpperCase();
      const maxLabelW = (maxX - minX) - 28;
      if (maxLabelW > 20) {
        while (labelText.length > 1 && this.ctx.measureText(labelText + '…').width > maxLabelW) {
          labelText = labelText.slice(0, -1);
        }
        if (this.ctx.measureText(zone.name.toUpperCase()).width > maxLabelW) labelText += '…';
      }
      const lx = minX + 8;
      const ly = minY + 8 + 3 + labelFontSize;
      // Background pill
      const tw = this.ctx.measureText(labelText).width;
      this.ctx.fillStyle = 'rgba(0,0,0,0.65)';
      this._roundRect(lx - 6, ly - labelFontSize - 3, tw + 12, labelFontSize + 6, 3);
      this.ctx.fill();
      // Text
      this.ctx.fillStyle = zone.enabled ? zone.color : '#888';
      this.ctx.fillText(labelText, lx, ly);

      // Vertex handles (only for selected zone)
      if (isSelected && zone.enabled) {
        canvasPts.forEach((pt, vIdx) => {
          const isHovered = this.hoveredHandle &&
            this.hoveredHandle.zoneIdx === zIdx &&
            this.hoveredHandle.vertexIdx === vIdx;
          const r = isHovered ? 10 : 7;
          this.ctx.beginPath();
          this.ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
          this.ctx.fillStyle = selfIntersecting ? 'rgba(255,34,34,0.9)' : 'rgba(255,255,255,0.9)';
          this.ctx.fill();
          this.ctx.strokeStyle = zone.color;
          this.ctx.lineWidth = 2;
          this.ctx.stroke();
        });
      }
    });

    // Self-intersection warning overlay
    const selZone = this._getDraftById(this.selectedZoneId);
    if (selZone && this._polygonSelfIntersects(selZone.polygon)) {
      this.ctx.fillStyle = 'rgba(255,34,34,0.85)';
      this.ctx.font = 'bold 13px JetBrains Mono, monospace';
      const warnText = '⚠ SELF-INTERSECTING — FIX BEFORE SAVING';
      const ww = this.ctx.measureText(warnText).width;
      this.ctx.fillStyle = 'rgba(255,0,0,0.15)';
      this._roundRect(cw / 2 - ww / 2 - 12, 12, ww + 24, 30, 6);
      this.ctx.fill();
      this.ctx.fillStyle = '#ff4444';
      this.ctx.fillText(warnText, cw / 2 - ww / 2, 33);
    }
  }

  _roundRect(x, y, w, h, r) {
    this.ctx.beginPath();
    this.ctx.moveTo(x + r, y);
    this.ctx.lineTo(x + w - r, y);
    this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    this.ctx.lineTo(x + w, y + h - r);
    this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    this.ctx.lineTo(x + r, y + h);
    this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    this.ctx.lineTo(x, y + r);
    this.ctx.quadraticCurveTo(x, y, x + r, y);
    this.ctx.closePath();
  }

  _hexToRGBA(hex, alpha) {
    if (!hex || hex.length !== 7) return `rgba(128,128,128,${alpha})`;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ────────────────────────────────────────────
  // Pointer events (vertex dragging)
  // ────────────────────────────────────────────

  _getCanvasPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return { x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
             y: (e.clientY - rect.top) * (this.canvas.height / rect.height) };
  }

  _findHandleAt(cx, cy) {
    // Only check selected zone
    const selIdx = this.draft.findIndex(z => z.id === this.selectedZoneId);
    if (selIdx < 0) return null;
    const zone = this.draft[selIdx];
    if (!zone.enabled) return null;
    for (let v = 0; v < zone.polygon.length; v++) {
      const pt = this._normToCanvas(zone.polygon[v][0], zone.polygon[v][1]);
      const dist = Math.sqrt((cx - pt.x) ** 2 + (cy - pt.y) ** 2);
      if (dist <= 12) return { zoneIdx: selIdx, vertexIdx: v };
    }
    return null;
  }

  _findZoneAt(cx, cy) {
    const [nx, ny] = this._canvasToNorm(cx, cy);
    // Find smallest-area zone containing the point
    let best = null;
    let bestArea = Infinity;
    for (let i = 0; i < this.draft.length; i++) {
      const zone = this.draft[i];
      if (!zone.enabled) continue;
      if (this._pointInPolygon(nx, ny, zone.polygon)) {
        const area = this._polygonArea(zone.polygon);
        if (area < bestArea) {
          bestArea = area;
          best = zone;
        }
      }
    }
    return best;
  }

  _onPointerDown(e) {
    const pos = this._getCanvasPos(e);

    // Check for handle drag (on selected zone)
    const handle = this._findHandleAt(pos.x, pos.y);
    if (handle) {
      const zone = this.draft[handle.zoneIdx];
      this.dragging = {
        zoneIdx: handle.zoneIdx,
        vertexIdx: handle.vertexIdx,
        startNorm: [...zone.polygon[handle.vertexIdx]],
      };
      this.canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
      return;
    }

    // Check for zone selection
    const zone = this._findZoneAt(pos.x, pos.y);
    if (zone) {
      this.selectedZoneId = zone.id;
    } else {
      this.selectedZoneId = null;
    }
    this._buildZoneList();
    this._buildPropsPanel();
    this.render();
  }

  _onPointerMove(e) {
    const pos = this._getCanvasPos(e);

    if (this.dragging) {
      const [nx, ny] = this._canvasToNorm(pos.x, pos.y);
      const zone = this.draft[this.dragging.zoneIdx];
      zone.polygon[this.dragging.vertexIdx] = [nx, ny];
      this.render();
      this._updatePropsInfo();
      return;
    }

    // Hover effect on handles
    const handle = this._findHandleAt(pos.x, pos.y);
    if (handle !== this.hoveredHandle) {
      this.hoveredHandle = handle;
      this.canvas.style.cursor = handle ? 'grab' : 'default';
      this.render();
    }
  }

  _onPointerUp(e) {
    if (this.dragging) {
      this.canvas.releasePointerCapture(e.pointerId);
      this.dragging = null;
      this._buildZoneList(); // update dirty indicator
    }
  }

  _onKeyDown(e) {
    if (!this.isOpen) return;
    if (e.key === 'Escape' && this.dragging) {
      // Revert drag
      const zone = this.draft[this.dragging.zoneIdx];
      zone.polygon[this.dragging.vertexIdx] = this.dragging.startNorm;
      this.dragging = null;
      this.render();
      e.preventDefault();
    }
  }

  _onResize() {
    if (!this.isOpen) return;
    this._resizeCanvas();
    this.render();
  }

  _onBeforeUnload(e) {
    if (this.isDirty() && !this.suppressBeforeUnload) {
      e.preventDefault();
      e.returnValue = '';
    }
  }

  // ────────────────────────────────────────────
  // Geometry helpers
  // ────────────────────────────────────────────

  _pointInPolygon(px, py, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const xi = polygon[i][0], yi = polygon[i][1];
      const xj = polygon[j][0], yj = polygon[j][1];
      const intersect = ((yi > py) !== (yj > py)) &&
        (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  _polygonArea(polygon) {
    const n = polygon.length;
    if (n < 3) return 0;
    let area = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      area += polygon[i][0] * polygon[j][1];
      area -= polygon[j][0] * polygon[i][1];
    }
    return Math.abs(area) / 2;
  }

  _segmentsIntersect(p1, p2, p3, p4) {
    function ccw(a, b, c) {
      return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0]);
    }
    const d1 = ccw(p3, p4, p1), d2 = ccw(p3, p4, p2);
    const d3 = ccw(p1, p2, p3), d4 = ccw(p1, p2, p4);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
    return false;
  }

  _polygonSelfIntersects(polygon) {
    const n = polygon.length;
    if (n < 4) return false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 2; j < n; j++) {
        if (i === 0 && j === n - 1) continue;
        if (this._segmentsIntersect(polygon[i], polygon[(i + 1) % n],
                                     polygon[j], polygon[(j + 1) % n])) return true;
      }
    }
    return false;
  }

  // ────────────────────────────────────────────
  // Zone list panel (left side list)
  // ────────────────────────────────────────────

  _buildZoneList() {
    if (!this.zoneListEl) return;
    this.zoneListEl.innerHTML = '';

    this.draft.forEach((zone) => {
      const item = document.createElement('div');
      item.className = 'ze-zone-item' + (zone.id === this.selectedZoneId ? ' selected' : '');
      if (!zone.enabled) item.classList.add('disabled');

      const dot = document.createElement('span');
      dot.className = 'ze-zone-dot';
      dot.style.background = zone.enabled ? zone.color : '#555';

      const name = document.createElement('span');
      name.className = 'ze-zone-name';
      name.textContent = zone.name + (!zone.enabled ? ' (Disabled)' : '');

      const verts = document.createElement('span');
      verts.className = 'ze-zone-verts';
      verts.textContent = `${zone.polygon.length} pts`;

      item.appendChild(dot);
      item.appendChild(name);
      item.appendChild(verts);

      item.addEventListener('click', () => {
        this.selectedZoneId = zone.id;
        this._buildZoneList();
        this._buildPropsPanel();
        this.render();
      });

      this.zoneListEl.appendChild(item);
    });

    // Disabled "Add Zone" placeholder
    const addBtn = document.createElement('button');
    addBtn.className = 'ze-btn ze-btn-add disabled';
    addBtn.textContent = '+ Add Zone';
    addBtn.title = 'Coming in Phase 4b';
    addBtn.disabled = true;
    this.zoneListEl.appendChild(addBtn);
  }

  // ────────────────────────────────────────────
  // Property panel
  // ────────────────────────────────────────────

  _buildPropsPanel() {
    if (!this.propsEl) return;
    const zone = this._getDraftById(this.selectedZoneId);
    if (!zone) {
      this.propsEl.innerHTML = '<div class="ze-props-empty">Select a zone to edit its properties.</div>';
      return;
    }

    this.propsEl.innerHTML = '';

    // ID (read-only)
    this.propsEl.appendChild(this._propRow('ID', () => {
      const span = document.createElement('span');
      span.className = 'ze-prop-readonly';
      span.textContent = zone.id;
      return span;
    }));

    // Name
    this.propsEl.appendChild(this._propRow('Name', () => {
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'ze-input';
      input.value = zone.name;
      input.maxLength = 64;
      input.addEventListener('input', () => {
        zone.name = input.value.trim() || zone.id;
        this._buildZoneList();
        this.render();
      });
      return input;
    }));

    // Color
    this.propsEl.appendChild(this._propRow('Color', () => {
      const input = document.createElement('input');
      input.type = 'color';
      input.className = 'ze-input-color';
      input.value = zone.color;
      input.addEventListener('input', () => {
        zone.color = input.value;
        this._buildZoneList();
        this.render();
      });
      return input;
    }));

    // Assignment Mode
    this.propsEl.appendChild(this._propRow('Assignment', () => {
      const sel = document.createElement('select');
      sel.className = 'ze-select';
      ['center', 'bottom_center'].forEach(val => {
        const opt = document.createElement('option');
        opt.value = val;
        opt.textContent = val === 'center' ? 'Center' : 'Bottom Center';
        if (val === zone.assignment_mode) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.addEventListener('change', () => {
        zone.assignment_mode = sel.value;
      });
      return sel;
    }));

    // Enabled toggle
    this.propsEl.appendChild(this._propRow('Enabled', () => {
      const label = document.createElement('label');
      label.className = 'ze-toggle-label';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = zone.enabled;
      cb.addEventListener('change', () => {
        zone.enabled = cb.checked;
        this._buildZoneList();
        this._buildPropsPanel();
        this.render();
      });
      const slider = document.createElement('span');
      slider.className = 'ze-toggle-slider';
      label.appendChild(cb);
      label.appendChild(slider);
      return label;
    }));

    // Info: vertices, coverage
    this._updatePropsInfo();

    // Delete placeholder
    const delBtn = document.createElement('button');
    delBtn.className = 'ze-btn ze-btn-delete disabled';
    delBtn.textContent = 'Delete Zone';
    delBtn.title = 'Coming in Phase 4b';
    delBtn.disabled = true;
    delBtn.style.marginTop = '16px';
    this.propsEl.appendChild(delBtn);
  }

  _updatePropsInfo() {
    const zone = this._getDraftById(this.selectedZoneId);
    if (!zone) return;
    let infoEl = document.getElementById('ze-props-info');
    if (!infoEl) {
      infoEl = document.createElement('div');
      infoEl.id = 'ze-props-info';
      infoEl.className = 'ze-props-info';
      this.propsEl.appendChild(infoEl);
    }
    const area = this._polygonArea(zone.polygon);
    const coveragePct = (area * 100).toFixed(1);
    infoEl.textContent = `${zone.polygon.length} vertices · ${coveragePct}% coverage`;
  }

  _propRow(label, buildControl) {
    const row = document.createElement('div');
    row.className = 'ze-prop-row';
    const lbl = document.createElement('label');
    lbl.className = 'ze-prop-label';
    lbl.textContent = label;
    row.appendChild(lbl);
    row.appendChild(buildControl());
    return row;
  }

  // ────────────────────────────────────────────
  // Save / Cancel / Reset
  // ────────────────────────────────────────────

  async save() {
    // Client-side validation
    const errors = this._validate();
    if (errors.length > 0) {
      this._showToast('Validation errors:\n' + errors.join('\n'), 'error');
      return false;
    }

    // Warn about analytics reset
    if (!this._saveConfirmed) {
      const ok = confirm(
        'Saving zones will reset analytics counters (entry/exit, footfall) for active sessions.\n\nProceed?'
      );
      if (!ok) return false;
      this._saveConfirmed = true; // don't ask again this session
    }

    // Build PUT payload
    const payload = {
      coordinate_system: 'normalized',
      definitions: this.draft.map(z => ({
        id: z.id,
        name: z.name,
        polygon: z.polygon.map(pt => [
          Math.round(pt[0] * 1e6) / 1e6,
          Math.round(pt[1] * 1e6) / 1e6,
        ]),
        color: z.color,
        assignment_mode: z.assignment_mode,
        enabled: z.enabled,
      })),
    };

    try {
      const res = await fetch('/api/zones', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.status === 422) {
        const err = await res.json();
        const msgs = err.detail?.errors || [err.detail?.message || 'Validation failed'];
        this._showToast('Server validation failed:\n' + msgs.join('\n'), 'error');
        return false;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        this._showToast(`Save failed (${res.status}): ${err.detail || 'Unknown error'}`, 'error');
        return false;
      }

      // Success — refresh store and update snapshot
      if (window.zoneStore) await window.zoneStore.refresh();
      this.savedSnapshot = JSON.stringify(this.draft);
      this._showToast('Zones saved successfully!', 'success');
      return true;
    } catch (e) {
      this._showToast('Network error: ' + e.message, 'error');
      return false;
    }
  }

  cancel() {
    if (!this.confirmDiscard()) return false;
    this._loadDraft();
    this.selectedZoneId = null;
    this._buildZoneList();
    this._buildPropsPanel();
    this.render();
    return true;
  }

  async resetToDefaults() {
    if (!confirm('Reset all zones to config.yaml defaults?\nThis will overwrite all custom zones.')) return;

    try {
      const res = await fetch('/api/zones/reset', { method: 'POST' });
      if (!res.ok) {
        this._showToast('Reset failed', 'error');
        return;
      }
      if (window.zoneStore) await window.zoneStore.refresh();
      this._loadDraft();
      this.selectedZoneId = null;
      this._buildZoneList();
      this._buildPropsPanel();
      this.render();
      this._showToast('Reset to defaults', 'success');
    } catch (e) {
      this._showToast('Network error: ' + e.message, 'error');
    }
  }

  _validate() {
    const errors = [];
    this.draft.forEach((z, i) => {
      if (!z.name || z.name.trim().length === 0) {
        errors.push(`Zone "${z.id}": Name cannot be empty.`);
      }
      if (z.name && z.name.length > 64) {
        errors.push(`Zone "${z.id}": Name too long (max 64 chars).`);
      }
      if (z.polygon.length < 3) {
        errors.push(`Zone "${z.id}": Needs at least 3 vertices.`);
      }
      for (const pt of z.polygon) {
        if (pt[0] < 0 || pt[0] > 1 || pt[1] < 0 || pt[1] > 1) {
          errors.push(`Zone "${z.id}": Vertex out of range (must be 0–1).`);
          break;
        }
      }
      if (this._polygonSelfIntersects(z.polygon)) {
        errors.push(`Zone "${z.id}": Polygon is self-intersecting.`);
      }
    });
    return errors;
  }

  // ────────────────────────────────────────────
  // Toast notifications
  // ────────────────────────────────────────────

  _showToast(msg, type = 'info') {
    if (!this.toastEl) return;
    this.toastEl.textContent = msg;
    this.toastEl.className = `ze-toast visible ${type}`;
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      this.toastEl.classList.remove('visible');
    }, type === 'error' ? 6000 : 3000);
  }
}

window.zoneEditor = new ZoneEditor();
