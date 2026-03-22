/**
 * Physics Web Worker.
 * Runs NBodySim in a dedicated thread. Communicates with main thread via
 * postMessage / transferable ArrayBuffers.
 *
 * Messages in:
 *   { type: 'init',     config, scenario, seed }
 *   { type: 'step',     count }
 *   { type: 'pause' }
 *   { type: 'resume' }
 *   { type: 'reset',    scenario, seed }
 *   { type: 'impulse',  bodyIdx, ix, iy, iz }
 *   { type: 'setConfig', cfg }
 *   { type: 'selfTest' }
 *
 * Messages out:
 *   { type: 'ready' }
 *   { type: 'state',    snapshot (with transferable buffers) }
 *   { type: 'selfTest', results }
 *   { type: 'error',    message }
 */

import { NBodySim } from '../physics/core/nbodysim.js';

let sim = null;
let running = false;
let loopId = null;

// Target physics rate: up to 4 substeps per 16ms render frame
const SUBSTEPS_PER_MESSAGE = 4;

self.onmessage = (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        sim = new NBodySim(msg.config ?? {});
        sim.reset(msg.scenario ?? 'headon', msg.seed ?? 42);
        running = false;
        postState();
        self.postMessage({ type: 'ready' });
        break;

      case 'step': {
        if (!sim) break;
        const count = msg.count ?? SUBSTEPS_PER_MESSAGE;
        const t0 = performance.now();
        for (let s = 0; s < count; s++) sim.stepPhysics();
        postState(performance.now() - t0);
        break;
      }

      case 'pause':
        running = false;
        break;

      case 'resume':
        running = true;
        scheduleNext();
        break;

      case 'reset':
        running = false;
        if (!sim) sim = new NBodySim({});
        sim.reset(msg.scenario ?? 'headon', msg.seed ?? sim._rng.seed);
        postState(0);
        break;

      case 'impulse':
        if (!sim) break;
        sim.applyImpulse(msg.bodyIdx, msg.ix, msg.iy, msg.iz);
        break;

      case 'slice':
        if (!sim) break;
        sim.sliceBody(msg.bodyIdx, msg.planePoint, msg.planeNormal);
        break;

      case 'setConfig':
        if (!sim) break;
        sim.updateConfig(msg.cfg);
        break;

      case 'selfTest':
        if (!sim) { self.postMessage({ type: 'selfTest', error: 'not initialized' }); break; }
        const results = sim.selfTestOrbitDrift(10000);
        self.postMessage({ type: 'selfTest', results });
        break;

      default:
        self.postMessage({ type: 'error', message: `Unknown message type: ${msg.type}` });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message, stack: err.stack });
  }
};

function scheduleNext() {
  if (!running) return;
  // Use setTimeout(0) to yield to message queue periodically
  setTimeout(tick, 0);
}

function tick() {
  if (!running || !sim) return;
  const t0 = performance.now();
  for (let s = 0; s < SUBSTEPS_PER_MESSAGE; s++) {
    sim.stepPhysics();
  }
  const elapsed = performance.now() - t0;
  postState(elapsed);
  // Adaptive yield: if step took too long, yield more
  const delay = elapsed > 10 ? 4 : 0;
  if (running) setTimeout(tick, delay);
}

function postState(physicsMs = 0) {
  if (!sim) return;
  const snap = sim.getSnapshot();
  snap.physicsMs = physicsMs;

  // Collect transferable buffers
  const transferables = [
    snap.positions.buffer,
    snap.velocities.buffer,
    snap.masses.buffer,
    snap.radii.buffer,
    snap.types.buffer,
    snap.colors.buffer,
    snap.stresses.buffer,
    snap.ids.buffer,
  ];

  if (snap.sph) {
    if (snap.sph.positions) transferables.push(snap.sph.positions.buffer);
    if (snap.sph.alphas)    transferables.push(snap.sph.alphas.buffer);
    if (snap.sph.temps)     transferables.push(snap.sph.temps.buffer);
  }

  self.postMessage({ type: 'state', snapshot: snap }, transferables);
}
