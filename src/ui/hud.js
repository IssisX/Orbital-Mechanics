/**
 * HUD — All UI controls, sliders, toggles, profiler, scenario presets,
 * export/import, keyboard map, drag-impulse tool, and slice tool.
 *
 * Pure DOM/CSS — no framework dependencies.
 */

export class HUD {
  /**
   * @param {HTMLElement} root     Container element
   * @param {Object}      sim      Physics worker proxy (has .send(msg))
   * @param {Object}      renderer  WebGL2/WebGPU renderer instance
   * @param {Object}      profiler  Profiler instance
   * @param {Object}      opts     Initial config values
   */
  constructor(root, sim, renderer, profiler, opts = {}) {
    this._root     = root;
    this._sim      = sim;
    this._renderer = renderer;
    this._profiler = profiler;
    this._opts     = opts;

    this._paused   = false;
    this._seed     = opts.seed ?? 42;
    this._scenario = opts.scenario ?? 'headon';
    this._activeTool = 'orbit'; // 'orbit' | 'impulse' | 'slice'

    this._slicePath = []; // screen points for slice tool

    this._panel = null;
    this._controls = {};  // name → element
    this._buildUI();
    this._attachGlobalKeyboard();
  }

  // ─── Build ────────────────────────────────────────────────────────────────

  _buildUI() {
    const panel = document.createElement('div');
    panel.id = 'hud-panel';
    panel.innerHTML = this._templateHTML();
    this._root.appendChild(panel);
    this._panel = panel;

    this._bindControls();
    this._updateSeedDisplay();
  }

  _templateHTML() {
    return `
<div id="hud-inner">
  <!-- Header -->
  <div class="hud-header">
    <span class="hud-title">⊛ Celestial Collider</span>
    <button id="btn-hide-hud" title="Hide panel (H)">▶</button>
  </div>

  <!-- Simulation group -->
  <details open class="hud-group">
    <summary>Simulation</summary>
    <div class="hud-row">
      <button id="btn-play" class="btn-primary">▶ Play</button>
      <button id="btn-step">⏩ Step</button>
      <button id="btn-reset">↺ Reset</button>
    </div>
    <div class="hud-row">
      <label>Scenario</label>
      <select id="sel-scenario">
        <option value="headon">Head-On Impact</option>
        <option value="grazing">Grazing Impact</option>
        <option value="roche">Roche Skim</option>
        <option value="tribody">Tri-Body Dance</option>
        <option value="ring">Debris Ring</option>
      </select>
    </div>
    <div class="hud-row">
      <label>Seed</label>
      <input id="inp-seed" type="number" min="0" max="999999" value="42" style="width:80px">
      <button id="btn-reset-seed">Reset with Seed</button>
    </div>
    <div class="hud-row seed-display">
      <span>Current seed: <b id="lbl-seed">42</b></span>
    </div>
    <div class="hud-row">
      <button id="btn-export">⬆ Export JSON</button>
      <button id="btn-import">⬇ Import JSON</button>
      <input id="inp-import-file" type="file" accept=".json" style="display:none">
    </div>
  </details>

  <!-- Physics group -->
  <details open class="hud-group">
    <summary>Physics</summary>
    ${this._slider('sl-theta',   'Theta (BH)',      0.1, 1.5, 0.6,  0.05)}
    ${this._slider('sl-epsilon', 'Softening ε',     0.001, 0.2, 0.03, 0.001)}
    ${this._slider('sl-dtmin',   'dt min',          1e-6, 1e-3, 1e-5, 1e-6)}
    ${this._slider('sl-dtmax',   'dt max',          1e-4, 0.1,  0.02, 1e-4)}
    ${this._slider('sl-cfl',     'CFL factor',      0.1,  1.0,  0.4,  0.05)}
    ${this._slider('sl-restitution','Restitution',  0.0,  1.0,  0.3,  0.05)}
    ${this._slider('sl-cohesion','Cohesion',        0.01, 2.0,  0.5,  0.01)}
  </details>

  <!-- Fracture group -->
  <details class="hud-group">
    <summary>Fracture</summary>
    ${this._slider('sl-pde',    'Modulus E',  100, 50000, 5000, 100)}
    ${this._slider('sl-pdsc',   'Crit. stretch sc', 0.01, 1.0, 0.25, 0.01)}
    ${this._slider('sl-pdzeta','Damping ζ',  0.0,  0.5,  0.05, 0.01)}
    ${this._slider('sl-sphh',  'SPH radius h', 0.3, 5.0, 1.5, 0.1)}
    ${this._slider('sl-sphk',  'SPH pressure k', 0.1, 10.0, 2.0, 0.1)}
  </details>

  <!-- Rendering group -->
  <details class="hud-group">
    <summary>Rendering</summary>
    <div class="hud-row">
      <label>Renderer</label>
      <span id="lbl-renderer" class="badge">WebGL2</span>
    </div>
    ${this._toggle('tog-bonds',   'Show Bonds',  false)}
    ${this._toggle('tog-octree',  'Show Octree', false)}
    ${this._toggle('tog-bvh',     'Show BVH',    false)}
    ${this._toggle('tog-energy',  'Energy Error',true)}
    ${this._toggle('tog-bloom',   'Bloom',       false)}
    ${this._toggle('tog-cinematic','Cinematic',  false)}
  </details>

  <!-- Debug group -->
  <details class="hud-group">
    <summary>Debug</summary>
    <div class="hud-row">
      <button id="btn-selftest">▷ Run Self-Test</button>
    </div>
    <div id="selftest-result" class="hud-monospace" style="min-height:32px;font-size:10px;padding:4px;"></div>
    <div class="hud-row">
      <button id="btn-stress-many">Stress: Many Bodies</button>
    </div>
    <div class="hud-row">
      <button id="btn-stress-sph">Stress: SPH Heavy</button>
    </div>
    <div class="hud-row">
      <button id="btn-stress-shatter">Stress: Shatter</button>
    </div>
    ${this._toggle('tog-profiler','Show Profiler', true)}
  </details>

  <!-- Tools -->
  <details open class="hud-group">
    <summary>Tools</summary>
    <div class="hud-row tool-row">
      <button id="tool-orbit"   class="tool-btn active" title="Q">🔭 Orbit</button>
      <button id="tool-impulse" class="tool-btn"        title="I">💥 Impulse</button>
      <button id="tool-slice"   class="tool-btn"        title="S">✂ Slice</button>
    </div>
    <div class="hud-hint" id="tool-hint">Orbit camera active</div>
  </details>

  <!-- Controls reference -->
  <details class="hud-group">
    <summary>Controls</summary>
    <div class="hud-monospace" style="font-size:10px;line-height:1.7">
      <b>Camera:</b><br>
      Left drag: Orbit<br>
      Right drag: Pan<br>
      Scroll: Dolly<br>
      Pinch (mobile): Dolly<br>
      <b>Keyboard:</b><br>
      Space: Play/Pause<br>
      R: Reset<br>
      H: Hide HUD<br>
      Q: Orbit tool<br>
      I: Impulse tool<br>
      S: Slice tool<br>
      P: Profiler toggle<br>
      B: Toggle Bloom<br>
    </div>
  </details>
</div>
    `;
  }

  _slider(id, label, min, max, value, step) {
    return `
    <div class="hud-row hud-slider-row">
      <label for="${id}">${label}</label>
      <input type="range" id="${id}" min="${min}" max="${max}" value="${value}" step="${step}">
      <span id="${id}-val">${value}</span>
    </div>`;
  }

  _toggle(id, label, def) {
    return `
    <div class="hud-row">
      <label for="${id}">${label}</label>
      <input type="checkbox" id="${id}" ${def ? 'checked' : ''}>
    </div>`;
  }

  // ─── Style ────────────────────────────────────────────────────────────────

  injectStyle() {
    if (document.getElementById('hud-style')) return;
    const style = document.createElement('style');
    style.id = 'hud-style';
    style.textContent = `
      #hud-panel {
        position: fixed;
        top: 0; right: 0;
        width: 270px;
        height: 100vh;
        overflow-y: auto;
        background: rgba(5, 8, 20, 0.92);
        border-left: 1px solid rgba(100,180,255,0.15);
        color: #d0e8ff;
        font: 12px/1.5 'Inter', 'Segoe UI', system-ui, sans-serif;
        z-index: 100;
        scrollbar-width: thin;
        scrollbar-color: rgba(80,160,255,0.3) transparent;
        transition: transform 0.25s;
        user-select: none;
      }
      #hud-panel.hidden { transform: translateX(270px); }
      #hud-inner { padding: 8px 10px 24px; }
      .hud-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 6px 0 8px;
        border-bottom: 1px solid rgba(100,180,255,0.15);
        margin-bottom: 6px;
      }
      .hud-title { font-size: 13px; font-weight: 700; color: #80ccff; letter-spacing: .5px; }
      #btn-hide-hud {
        background: none; border: 1px solid rgba(100,180,255,0.25);
        color: #80ccff; cursor: pointer; border-radius: 3px; padding: 2px 6px; font-size:11px;
      }
      .hud-group {
        border: 1px solid rgba(100,180,255,0.1);
        border-radius: 5px; margin-bottom: 7px; overflow: hidden;
      }
      .hud-group summary {
        padding: 5px 8px;
        background: rgba(30,60,120,0.4);
        cursor: pointer;
        font-size: 11px;
        font-weight: 600;
        color: #a0d4ff;
        list-style: none;
        display: flex; align-items: center; gap: 5px;
      }
      .hud-group summary::before { content: '▸'; font-size: 9px; }
      .hud-group[open] summary::before { content: '▾'; }
      .hud-row {
        display: flex; align-items: center; gap: 6px;
        padding: 3px 8px; flex-wrap: wrap;
      }
      .hud-row label { flex: 0 0 85px; font-size: 11px; color: #90b8e0; }
      .hud-slider-row input[type=range] { flex: 1; min-width: 60px; }
      .hud-slider-row span { font-size: 10px; min-width: 36px; color: #c0e0ff; text-align:right; }
      button, select, input[type=number] {
        background: rgba(30,60,120,0.5);
        border: 1px solid rgba(100,180,255,0.25);
        color: #d0e8ff; border-radius: 4px;
        padding: 3px 7px; font-size: 11px; cursor: pointer;
        transition: background 0.15s;
      }
      button:hover { background: rgba(60,120,220,0.5); }
      .btn-primary { background: rgba(40,100,200,0.6); }
      input[type=checkbox] { width: 14px; height: 14px; cursor: pointer; }
      input[type=range] { -webkit-appearance: none; height: 4px;
        background: rgba(100,180,255,0.2); border-radius: 2px; border: none;
        outline: none; cursor: pointer; }
      input[type=range]::-webkit-slider-thumb { -webkit-appearance: none;
        width: 12px; height: 12px; background: #4499ff; border-radius: 50%; cursor: pointer; }
      .badge { background: rgba(40,100,200,0.4); border: 1px solid rgba(100,180,255,0.3);
        border-radius: 4px; padding: 2px 6px; font-size: 10px; color: #aaddff; }
      .tool-row { gap: 4px; }
      .tool-btn {
        flex: 1; text-align: center; padding: 4px 2px; font-size: 11px;
        background: rgba(20,40,80,0.6);
      }
      .tool-btn.active { background: rgba(40,100,200,0.6); border-color: rgba(100,180,255,0.5); }
      .hud-hint { font-size: 10px; color: #6090c0; padding: 2px 8px 4px; }
      .hud-monospace { font-family: monospace; color: #90c8e0; padding: 2px 8px; }
      .seed-display { font-size: 11px; color: #70a0c0; }
      select { padding: 3px 4px; }
    `;
    document.head.appendChild(style);
  }

  // ─── Bind ─────────────────────────────────────────────────────────────────

  _bindControls() {
    const p = this._panel;
    const get = id => p.querySelector('#' + id);

    // Play/Pause
    const btnPlay = get('btn-play');
    btnPlay.addEventListener('click', () => this._togglePlay());

    // Step
    get('btn-step').addEventListener('click', () => {
      this._sim.send({ type: 'step', count: 1 });
    });

    // Reset
    get('btn-reset').addEventListener('click', () => this._doReset());
    get('btn-reset-seed').addEventListener('click', () => {
      this._seed = parseInt(get('inp-seed').value) || 42;
      this._doReset();
    });

    // Scenario
    const selScene = get('sel-scenario');
    selScene.value = this._scenario;
    selScene.addEventListener('change', () => {
      this._scenario = selScene.value;
      this._doReset();
    });

    // Sliders
    const sliders = [
      ['sl-theta',       'theta',       v => +v],
      ['sl-epsilon',     'epsilon',     v => +v],
      ['sl-dtmin',       'dtMin',       v => +v],
      ['sl-dtmax',       'dtMax',       v => +v],
      ['sl-cfl',         'cfl',         v => +v],
      ['sl-restitution', 'restitution', v => +v],
      ['sl-cohesion',    'cohesion',    v => +v],
      ['sl-pde',         'pdE',         v => +v],
      ['sl-pdsc',        'pdSc',        v => +v],
      ['sl-pdzeta',      'pdZeta',      v => +v],
      ['sl-sphh',        'sphH',        v => +v],
      ['sl-sphk',        'sphK',        v => +v],
    ];
    for (const [id, key, parse] of sliders) {
      const el = get(id);
      if (!el) continue;
      const lbl = get(id + '-val');
      el.addEventListener('input', () => {
        const v = parse(el.value);
        if (lbl) lbl.textContent = v.toPrecision(3);
        this._sim.send({ type: 'setConfig', cfg: { [key]: v } });
      });
    }

    // Toggles
    const togBonds = get('tog-bonds');
    togBonds?.addEventListener('change', () => {
      if (this._renderer) this._renderer.showBonds = togBonds.checked;
    });

    const togBloom = get('tog-bloom');
    togBloom?.addEventListener('change', () => {
      this._renderer?.setBloom(togBloom.checked);
    });

    const togCinematic = get('tog-cinematic');
    togCinematic?.addEventListener('change', () => {
      const on = togCinematic.checked;
      this._renderer?.setBloom(on);
      if (this._renderer?.renderer) {
        this._renderer.renderer.toneMappingExposure = on ? 1.6 : 1.25;
      }
    });

    const togProfiler = get('tog-profiler');
    togProfiler?.addEventListener('change', () => {
      this._profiler?.setVisible(togProfiler.checked);
    });

    get('tog-octree')?.addEventListener('change', (e) => {
      if (this._renderer) this._renderer.showOctree = e.target.checked;
    });
    get('tog-bvh')?.addEventListener('change', (e) => {
      if (this._renderer) this._renderer.showBVH = e.target.checked;
    });

    // Hide HUD button
    get('btn-hide-hud').addEventListener('click', () => this.toggleHUD());

    // Self-test
    get('btn-selftest').addEventListener('click', () => {
      const el = get('selftest-result');
      el.textContent = 'Running…';
      this._sim.send({ type: 'selfTest' });
    });

    // Stress scenarios
    get('btn-stress-many')?.addEventListener('click', () => {
      this._scenario = 'ring';
      this._doReset();
    });
    get('btn-stress-sph')?.addEventListener('click', () => {
      this._scenario = 'headon';
      this._doReset();
    });
    get('btn-stress-shatter')?.addEventListener('click', () => {
      this._scenario = 'headon';
      this._seed = 1234;
      this._doReset();
    });

    // Export / Import
    get('btn-export').addEventListener('click', () => this._exportScenario());
    get('btn-import').addEventListener('click', () => get('inp-import-file').click());
    get('inp-import-file').addEventListener('change', (e) => this._importScenario(e.target.files[0]));

    // Tools
    const toolBtns = {
      'tool-orbit':   'orbit',
      'tool-impulse': 'impulse',
      'tool-slice':   'slice',
    };
    for (const [id, tool] of Object.entries(toolBtns)) {
      const btn = get(id);
      btn?.addEventListener('click', () => this._setTool(tool));
    }
  }

  // ─── Actions ─────────────────────────────────────────────────────────────

  _togglePlay() {
    this._paused = !this._paused;
    const btn = this._panel.querySelector('#btn-play');
    if (this._paused) {
      this._sim.send({ type: 'pause' });
      if (btn) btn.textContent = '▶ Play';
    } else {
      this._sim.send({ type: 'resume' });
      if (btn) btn.textContent = '⏸ Pause';
    }
  }

  _doReset() {
    this._sim.send({ type: 'reset', scenario: this._scenario, seed: this._seed });
    this._updateSeedDisplay();
  }

  _setTool(tool) {
    this._activeTool = tool;
    this._slicePath = [];
    const hints = {
      orbit:   'Orbit camera active',
      impulse: 'Click a body and drag to apply impulse',
      slice:   'Draw a line across a body to slice it',
    };
    const hintEl = this._panel.querySelector('#tool-hint');
    if (hintEl) hintEl.textContent = hints[tool] ?? '';

    for (const id of ['tool-orbit', 'tool-impulse', 'tool-slice']) {
      const btn = this._panel.querySelector('#' + id);
      btn?.classList.toggle('active', id === 'tool-' + tool);
    }

    // Disable orbit controls when using tools
    if (this._renderer?.controls) {
      this._renderer.controls.enabled = (tool === 'orbit');
    }
  }

  _updateSeedDisplay() {
    const el = this._panel?.querySelector('#lbl-seed');
    if (el) el.textContent = this._seed;
  }

  setRendererLabel(label) {
    const el = this._panel?.querySelector('#lbl-renderer');
    if (el) el.textContent = label;
  }

  displaySelfTestResult(results) {
    const el = this._panel?.querySelector('#selftest-result');
    if (!el) return;
    if (results.error) { el.textContent = '❌ ' + results.error; return; }
    const r = results.passed
      ? `✅ Orbit drift: ${results.driftPercent.toFixed(4)}% (< 0.5% ✓)`
      : `❌ Orbit drift: ${results.driftPercent.toFixed(4)}% (FAIL)`;
    el.textContent = r;
  }

  // ─── Export / Import ─────────────────────────────────────────────────────

  _exportScenario() {
    const data = {
      scenario: this._scenario,
      seed: this._seed,
      version: 1,
      timestamp: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `celestial-collider-${Date.now()}.json`;
    a.click();
  }

  _importScenario(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.scenario) this._scenario = data.scenario;
        if (data.seed !== undefined) this._seed = data.seed;
        this._panel.querySelector('#inp-seed').value = this._seed;
        this._panel.querySelector('#sel-scenario').value = this._scenario;
        this._doReset();
      } catch (err) {
        alert('Import failed: ' + err.message);
      }
    };
    reader.readAsText(file);
  }

  // ─── Pointer tool handling ────────────────────────────────────────────────

  /**
   * Called from main.js on pointerdown.
   * @param {number} ndcX,ndcY  Normalized device coords
   * @param {Object} snapshot   Latest simulation snapshot
   */
  onPointerDown(ndcX, ndcY, snapshot) {
    if (this._activeTool === 'impulse') {
      // Find nearest body
      const ai = this._renderer?.pickBody(ndcX, ndcY, snapshot);
      if (ai !== null && ai >= 0) {
        this._impulseTarget = { ai, startNdcX: ndcX, startNdcY: ndcY };
      }
    } else if (this._activeTool === 'slice') {
      this._slicePath = [{ x: ndcX, y: ndcY }];
    }
  }

  onPointerMove(ndcX, ndcY, snapshot) {
    if (this._activeTool === 'slice' && this._slicePath.length > 0) {
      this._slicePath.push({ x: ndcX, y: ndcY });
    }
  }

  onPointerUp(ndcX, ndcY, snapshot) {
    if (this._activeTool === 'impulse' && this._impulseTarget) {
      const { ai, startNdcX, startNdcY } = this._impulseTarget;
      const bodyId = snapshot.ids[ai];
      // Impulse proportional to drag distance
      const dx = (ndcX - startNdcX) * snapshot.radii[ai] * 5;
      const dy = (ndcY - startNdcY) * snapshot.radii[ai] * 5;
      this._sim.send({ type: 'impulse', bodyIdx: bodyId, ix: dx, iy: -dy, iz: 0 });
      this._impulseTarget = null;
    }

    if (this._activeTool === 'slice' && this._slicePath.length >= 2) {
      // Use first and last points to define slice plane
      const p0 = this._slicePath[0];
      const p1 = this._slicePath[this._slicePath.length - 1];

      // Find body intersected by the slice line
      const ai = this._renderer?.pickBody(
        (p0.x + p1.x) * 0.5, (p0.y + p1.y) * 0.5, snapshot
      );
      if (ai >= 0 && snapshot) {
        const bodyId = snapshot.ids[ai];
        // Build world-space plane from the camera's view
        const ray0 = this._renderer?.ndcToWorldRay(p0.x, p0.y);
        const ray1 = this._renderer?.ndcToWorldRay(p1.x, p1.y);
        if (ray0 && ray1) {
          // Plane normal = cross(ray0.dir, ray1.dir)
          const d0 = ray0.dir, d1 = ray1.dir;
          const nx = d0[1]*d1[2] - d0[2]*d1[1];
          const ny = d0[2]*d1[0] - d0[0]*d1[2];
          const nz = d0[0]*d1[1] - d0[1]*d1[0];
          const nl = Math.sqrt(nx*nx+ny*ny+nz*nz);

          this._sim.send({
            type: 'slice',
            bodyIdx: bodyId,
            planePoint: [snapshot.positions[ai*3], snapshot.positions[ai*3+1], snapshot.positions[ai*3+2]],
            planeNormal: [nx/nl, ny/nl, nz/nl],
          });
        }
      }
      this._slicePath = [];
    }
  }

  /** Draw slice overlay on a 2D canvas overlay */
  drawSliceOverlay(ctx, canvasW, canvasH) {
    if (this._slicePath.length < 2) return;
    ctx.clearRect(0, 0, canvasW, canvasH);
    ctx.strokeStyle = '#ff4444';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    for (let i = 0; i < this._slicePath.length; i++) {
      const p = this._slicePath[i];
      const x = (p.x * 0.5 + 0.5) * canvasW;
      const y = (0.5 - p.y * 0.5) * canvasH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ─── Keyboard ─────────────────────────────────────────────────────────────

  _attachGlobalKeyboard() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      switch (e.code) {
        case 'Space':      e.preventDefault(); this._togglePlay(); break;
        case 'KeyR':       this._doReset(); break;
        case 'KeyH':       this.toggleHUD(); break;
        case 'KeyQ':       this._setTool('orbit'); break;
        case 'KeyI':       this._setTool('impulse'); break;
        case 'KeyS':       this._setTool('slice'); break;
        case 'KeyP': {
          const tog = this._panel?.querySelector('#tog-profiler');
          if (tog) { tog.checked = !tog.checked; tog.dispatchEvent(new Event('change')); }
          break;
        }
        case 'KeyB': {
          const tog = this._panel?.querySelector('#tog-bloom');
          if (tog) { tog.checked = !tog.checked; tog.dispatchEvent(new Event('change')); }
          break;
        }
      }
    });
  }

  toggleHUD() {
    this._panel.classList.toggle('hidden');
    const btn = this._panel.querySelector('#btn-hide-hud');
    if (btn) btn.textContent = this._panel.classList.contains('hidden') ? '◀' : '▶';
  }

  get isPaused() { return this._paused; }
  get activeTool() { return this._activeTool; }
}
