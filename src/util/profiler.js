/**
 * Lightweight in-app profiler overlay.
 * Tracks frame time, physics step time, draw calls, bodies, fragments,
 * bonds, memory. Renders rolling line graphs on a canvas overlay.
 */
export class Profiler {
  constructor(options = {}) {
    this.historyLength = options.historyLength ?? 120;
    this.visible = true;

    // Rolling history buffers
    this._metrics = {
      frameMs:      new RingBuffer(this.historyLength),
      physicsMs:    new RingBuffer(this.historyLength),
      renderMs:     new RingBuffer(this.historyLength),
      drawCalls:    new RingBuffer(this.historyLength),
      bodies:       new RingBuffer(this.historyLength),
      fragments:    new RingBuffer(this.historyLength),
      bonds:        new RingBuffer(this.historyLength),
      sphParticles: new RingBuffer(this.historyLength),
      energyError:  new RingBuffer(this.historyLength),
    };

    // Current sample
    this._cur = {
      frameMs: 0, physicsMs: 0, renderMs: 0,
      drawCalls: 0, bodies: 0, fragments: 0,
      bonds: 0, sphParticles: 0, energyError: 0,
      memoryMB: 0,
    };

    this._frameStart   = performance.now();
    this._physicsStart = 0;
    this._renderStart  = 0;
    this._fps          = 0;
    this._fpsCount     = 0;
    this._fpsTimer     = 0;

    this._container = null;
    this._canvas    = null;
    this._ctx       = null;
    this._infoDiv   = null;
    this._mounted   = false;
  }

  // ─── Timing ────────────────────────────────────────────────────────────────

  frameBegin() {
    const now = performance.now();
    this._cur.frameMs = now - this._frameStart;
    this._frameStart  = now;
    this._fpsCount++;
    this._fpsTimer += this._cur.frameMs;
    if (this._fpsTimer >= 500) {
      this._fps      = Math.round(this._fpsCount * 1000 / this._fpsTimer);
      this._fpsCount = 0;
      this._fpsTimer = 0;
    }
  }

  physicsBegin() { this._physicsStart = performance.now(); }
  physicsEnd()   { this._cur.physicsMs = performance.now() - this._physicsStart; }
  renderBegin()  { this._renderStart  = performance.now(); }
  renderEnd()    { this._cur.renderMs  = performance.now() - this._renderStart; }

  // ─── Counters ──────────────────────────────────────────────────────────────

  setDrawCalls(n)    { this._cur.drawCalls    = n; }
  setBodies(n)       { this._cur.bodies       = n; }
  setFragments(n)    { this._cur.fragments    = n; }
  setBonds(n)        { this._cur.bonds        = n; }
  setSPHParticles(n) { this._cur.sphParticles = n; }
  setEnergyError(e)  { this._cur.energyError  = e; }
  setMemoryMB(m)     { this._cur.memoryMB     = m; }

  /** Push current sample into history and update DOM. */
  commit() {
    const m = this._metrics;
    const c = this._cur;
    m.frameMs.push(c.frameMs);
    m.physicsMs.push(c.physicsMs);
    m.renderMs.push(c.renderMs);
    m.drawCalls.push(c.drawCalls);
    m.bodies.push(c.bodies);
    m.fragments.push(c.fragments);
    m.bonds.push(c.bonds);
    m.sphParticles.push(c.sphParticles);
    m.energyError.push(c.energyError);
    this._updateDOM();
  }

  // ─── DOM ───────────────────────────────────────────────────────────────────

  mount(parentEl) {
    this._container = document.createElement('div');
    this._container.id = 'profiler-overlay';
    this._container.style.cssText = `
      position:absolute; top:8px; left:8px; z-index:9999;
      background:rgba(0,0,0,0.75); border:1px solid rgba(255,255,255,0.12);
      border-radius:6px; padding:8px 10px; font:11px/1.4 monospace;
      color:#e0e0e0; min-width:220px; pointer-events:none; user-select:none;
    `;

    this._infoDiv = document.createElement('div');
    this._container.appendChild(this._infoDiv);

    this._canvas = document.createElement('canvas');
    this._canvas.width = 220; this._canvas.height = 64;
    this._canvas.style.cssText = 'display:block; margin-top:6px; border-top:1px solid rgba(255,255,255,0.1);';
    this._container.appendChild(this._canvas);
    this._ctx = this._canvas.getContext('2d');

    parentEl.appendChild(this._container);
    this._mounted = true;
  }

  _updateDOM() {
    if (!this._mounted) return;
    const c = this._cur;
    const fps = this._fps;
    const warnF = c.frameMs > 20 ? 'color:#ff8844' : '';
    const warnP = c.physicsMs > 12 ? 'color:#ffcc44' : '';

    this._infoDiv.innerHTML =
      `<div style="font-weight:bold;color:#80d4ff;margin-bottom:3px">⊙ PROFILER</div>` +
      `<div style="${warnF}">Frame: ${c.frameMs.toFixed(1)} ms &nbsp; FPS: ${fps}</div>` +
      `<div style="${warnP}">Physics: ${c.physicsMs.toFixed(1)} ms</div>` +
      `<div>Render:  ${c.renderMs.toFixed(1)} ms</div>` +
      `<div>Draw calls: ${c.drawCalls}</div>` +
      `<div>Bodies: ${c.bodies} &nbsp; Frags: ${c.fragments}</div>` +
      `<div>Bonds: ${c.bonds} &nbsp; SPH: ${c.sphParticles}</div>` +
      `<div>Energy err: ${(c.energyError * 100).toFixed(3)}%</div>` +
      `<div>Mem est: ${c.memoryMB.toFixed(1)} MB</div>`;

    this._drawGraphs();
  }

  _drawGraphs() {
    const ctx = this._ctx;
    if (!ctx) return;
    const W = 220, H = 64, halfH = 32;
    ctx.clearRect(0, 0, W, H);

    const frameData   = this._metrics.frameMs.toArray();
    const physicsData = this._metrics.physicsMs.toArray();

    this._drawGraph(ctx, frameData,   0,     0, W, halfH, '#44aaff', 50, 16.67, 'frame ms');
    this._drawGraph(ctx, physicsData, 0, halfH, W, halfH, '#ffaa44', 30, 12,    'physics ms');
  }

  _drawGraph(ctx, data, x, y, w, h, color, yMax, threshold, label) {
    if (!data || data.length < 2) return;
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();

    // Threshold line
    if (threshold) {
      ctx.strokeStyle = 'rgba(255,80,80,0.4)';
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      const ty = y + h - (threshold / yMax) * h;
      ctx.beginPath(); ctx.moveTo(x, ty); ctx.lineTo(x + w, ty); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.beginPath();
    const n = data.length;
    for (let i = 0; i < n; i++) {
      const px = x + (i / (n - 1)) * w;
      const py = y + h - Math.min(data[i] / yMax, 1) * h;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.font = '9px monospace';
    ctx.fillText(label, x + 2, y + 10);
    ctx.restore();
  }

  setVisible(v) {
    this.visible = v;
    if (this._container) this._container.style.display = v ? '' : 'none';
  }

  getSnapshot() { return { ...this._cur, fps: this._fps }; }
}

// ─── Ring Buffer ───────────────────────────────────────────────────────────────

class RingBuffer {
  constructor(cap) {
    this._cap  = cap;
    this._buf  = new Float32Array(cap);
    this._head = 0;
    this._size = 0;
  }

  push(v) {
    this._buf[this._head] = v;
    this._head = (this._head + 1) % this._cap;
    if (this._size < this._cap) this._size++;
  }

  /** Return array ordered oldest→newest (allocates each call, used only for DOM update) */
  toArray() {
    const arr = new Float32Array(this._size);
    const start = (this._head - this._size + this._cap) % this._cap;
    for (let i = 0; i < this._size; i++) {
      arr[i] = this._buf[(start + i) % this._cap];
    }
    return arr;
  }

  get latest() { return this._buf[(this._head - 1 + this._cap) % this._cap]; }
  get size()   { return this._size; }
}
