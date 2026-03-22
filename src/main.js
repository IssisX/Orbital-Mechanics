/**
 * Celestial Collider — Main Application Bootstrap
 *
 * 1. Feature-detect WebGPU; fall back to WebGL2.
 * 2. Spin up the physics worker.
 * 3. Create renderer (WebGL2 or WebGPU delegate).
 * 4. Build HUD and profiler.
 * 5. Run the requestAnimationFrame loop with fixed physics substeps.
 * 6. Route pointer events → HUD tools.
 * 7. Self-test on first load.
 */

import { RendererWebGL2 }              from './render/renderer-webgl2.js';
import { RendererWebGPU, requestWebGPUDevice } from './render/renderer-webgpu.js';
import { Profiler }                    from './util/profiler.js';
import { HUD }                         from './ui/hud.js';

// ─── Canvas ───────────────────────────────────────────────────────────────────

const canvas = document.getElementById('main-canvas');

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width  = Math.floor(canvas.clientWidth  * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
}
resizeCanvas();
window.addEventListener('resize', () => {
  resizeCanvas();
  renderer?._onResize?.();
});

// ─── Overlay canvas (for slice tool) ─────────────────────────────────────────

const overlayCanvas = document.getElementById('overlay-canvas');
const overlayCtx = overlayCanvas?.getContext('2d');

function resizeOverlay() {
  if (!overlayCanvas) return;
  overlayCanvas.width  = canvas.clientWidth;
  overlayCanvas.height = canvas.clientHeight;
}
resizeOverlay();
window.addEventListener('resize', resizeOverlay);

// ─── Renderer ─────────────────────────────────────────────────────────────────

let renderer;
let usingWebGPU = false;

async function initRenderer() {
  const gpuDevice = await requestWebGPUDevice();
  if (gpuDevice) {
    try {
      const gpuRenderer = new RendererWebGPU(canvas, gpuDevice);
      await gpuRenderer.init();
      renderer = gpuRenderer;
      usingWebGPU = true;
      console.info('[CelestialCollider] WebGPU renderer active.');
      return 'WebGPU';
    } catch (e) {
      console.warn('[CelestialCollider] WebGPU init failed, falling back:', e);
    }
  }
  renderer = new RendererWebGL2(canvas);
  console.info('[CelestialCollider] WebGL2 renderer active.');
  return 'WebGL2';
}

// ─── Physics Worker ───────────────────────────────────────────────────────────

let latestSnapshot = null;
let physicsReady   = false;
const physWorker   = new Worker('./src/workers/physics.worker.js', { type: 'module' });

const simProxy = {
  send(msg) { physWorker.postMessage(msg); },
};

physWorker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'ready':
      physicsReady = true;
      profiler?.setBodies(0);
      // Start simulation
      simProxy.send({ type: 'resume' });
      break;

    case 'state':
      latestSnapshot = msg.snapshot;
      if (profiler) {
        if (msg.snapshot.physicsMs !== undefined) {
          profiler._cur.physicsMs = msg.snapshot.physicsMs;
        }
        profiler.setBodies(msg.snapshot.N);
        profiler.setFragments(
          msg.snapshot.types
            ? Array.from(msg.snapshot.types).filter(t => t >= 2).length
            : 0
        );
        const bonds = msg.snapshot.bondData
          ? Object.values(msg.snapshot.bondData).reduce((sum, d) =>
              sum + (d ? (d.active?.length ?? 0) / 6 + (d.broken?.length ?? 0) / 6 : 0), 0)
          : 0;
        profiler.setBonds(bonds | 0);
        profiler.setSPHParticles(msg.snapshot.sph?.count ?? 0);
        profiler.setEnergyError(msg.snapshot.energyError ?? 0);
        profiler.setMemoryMB(estimateMemory(msg.snapshot));
      }
      break;

    case 'selfTest':
      hud?.displaySelfTestResult(msg.results ?? { error: msg.error });
      break;

    case 'error':
      console.error('[PhysicsWorker]', msg.message, msg.stack ?? '');
      break;
  }
};

physWorker.onerror = (e) => {
  console.error('[PhysicsWorker] Fatal error:', e);
};

// ─── Profiler ────────────────────────────────────────────────────────────────

const profiler = new Profiler({ historyLength: 120 });

// ─── HUD ─────────────────────────────────────────────────────────────────────

let hud;

// ─── Memory estimate ─────────────────────────────────────────────────────────

function estimateMemory(snap) {
  if (!snap) return 0;
  let bytes = 0;
  if (snap.positions)  bytes += snap.positions.byteLength;
  if (snap.velocities) bytes += snap.velocities.byteLength;
  if (snap.masses)     bytes += snap.masses.byteLength;
  if (snap.sph?.positions) bytes += snap.sph.positions.byteLength;
  // Rough renderer overhead
  bytes += snap.N * 1024; // per-mesh geometry estimate
  return bytes / (1024 * 1024);
}

// ─── Render Loop ─────────────────────────────────────────────────────────────

let lastRenderTime = performance.now();

function loop(now) {
  requestAnimationFrame(loop);

  profiler.frameBegin();
  profiler.renderBegin();

  if (latestSnapshot && renderer) {
    try {
      renderer.render(latestSnapshot);

      // Draw stats
      const stats = renderer.getStats?.() ?? {};
      profiler.setDrawCalls(stats.drawCalls ?? 0);
    } catch (err) {
      console.error('[Render]', err);
    }
  }

  // Draw slice overlay
  if (overlayCtx && hud && hud.activeTool === 'slice') {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    hud.drawSliceOverlay(overlayCtx, overlayCanvas.width, overlayCanvas.height);
  } else if (overlayCtx) {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  profiler.renderEnd();
  profiler.commit();

  lastRenderTime = now;
}

// ─── Pointer Events ───────────────────────────────────────────────────────────

function canvasToNDC(e) {
  const rect = canvas.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
  const y = -((e.clientY - rect.top)  / rect.height) *  2 + 1;
  return { x, y };
}

let pointerDown = false;

canvas.addEventListener('pointerdown', (e) => {
  if (!hud || !latestSnapshot) return;
  const { x, y } = canvasToNDC(e);
  pointerDown = true;
  hud.onPointerDown(x, y, latestSnapshot);
});

canvas.addEventListener('pointermove', (e) => {
  if (!pointerDown || !hud || !latestSnapshot) return;
  const { x, y } = canvasToNDC(e);
  hud.onPointerMove(x, y, latestSnapshot);
});

canvas.addEventListener('pointerup', (e) => {
  if (!hud || !latestSnapshot) return;
  const { x, y } = canvasToNDC(e);
  hud.onPointerUp(x, y, latestSnapshot);
  pointerDown = false;
});

// Prevent context menu on right-click (used for pan)
canvas.addEventListener('contextmenu', (e) => e.preventDefault());

// ─── Bootstrap ────────────────────────────────────────────────────────────────

async function bootstrap() {
  const rendererLabel = await initRenderer();

  // Profiler mount
  profiler.mount(document.getElementById('app-container') ?? document.body);

  // HUD
  const hudContainer = document.getElementById('app-container') ?? document.body;
  hud = new HUD(hudContainer, simProxy, renderer, profiler, {
    seed: 42, scenario: 'headon',
  });
  hud.injectStyle();
  hud.setRendererLabel(rendererLabel);

  // Init physics worker
  simProxy.send({
    type: 'init',
    config: {
      G: 1.0, theta: 0.6, epsilon: 0.03,
      dtMin: 1e-5, dtMax: 0.02, cfl: 0.4,
      restitution: 0.3, cohesion: 0.5,
      pdE: 5000, pdSc: 0.25, pdZeta: 0.05,
      sphH: 1.5, sphK: 2.0,
      seed: 42,
    },
    scenario: 'headon',
    seed: 42,
  });

  // Start render loop
  requestAnimationFrame(loop);

  // Run self-test after physics is ready
  const waitReady = setInterval(() => {
    if (physicsReady) {
      clearInterval(waitReady);
      setTimeout(() => {
        console.info('[CelestialCollider] Running initial self-test…');
        simProxy.send({ type: 'selfTest' });
      }, 2000);
    }
  }, 100);

  console.info('[CelestialCollider] Bootstrap complete. Renderer:', rendererLabel);
}

bootstrap().catch(err => {
  console.error('[CelestialCollider] Fatal bootstrap error:', err);
  document.body.innerHTML =
    `<div style="color:#ff4444;padding:40px;font-family:monospace">
      <b>Fatal Error:</b> ${err.message}<br><br>
      Please check the console for details.
    </div>`;
});
