/**
 * Bond-based Peridynamics fracture system.
 *
 * Each simulated body is represented by a cloud of sub-particles (core + crust).
 * Bonds connect nearby particles within a "horizon" radius delta.
 * Bond force: F = c * s * (y_ij / |y_ij|)  where s = (|y_ij| - L0) / L0
 * Bond breaks permanently when |s| > s_c (critical stretch).
 *
 * On fracture, broken bond clusters are emitted as fragmentation events.
 */

import { v3dist, v3len, v3len2, v3normalize, v3sub, v3addScaled } from '../../util/math.js';

// Body-type for peridynamics simulation
export const BODY_INTACT = 0;
export const BODY_DAMAGED = 1;
export const BODY_SHATTERED = 2;

export class PeridynamicsSystem {
  /**
   * @param {Object} options
   * @param {number} options.E      Elastic modulus proxy (stiffness of bonds)
   * @param {number} options.sc     Critical stretch (fracture threshold)
   * @param {number} options.zeta   Damping coefficient
   * @param {number} options.horizon Horizon radius relative to body radius
   */
  constructor(options = {}) {
    this.E    = options.E    ?? 1e4;
    this.sc   = options.sc   ?? 0.25;
    this.zeta = options.zeta ?? 0.05;
    this.horizonFactor = options.horizon ?? 0.35;

    // Map from bodyId → PeribodyState
    this._bodies = new Map();

    // Pending fracture events for the meshing worker
    this._fractureEvents = [];
  }

  // ─── API ─────────────────────────────────────────────────────────────────────

  /**
   * Register a body for peridynamic fracture simulation.
   * Generates sub-particles and bond graph.
   */
  registerBody(bodyId, cx, cy, cz, radius, mass, particleCount, rng) {
    particleCount = Math.max(10, Math.min(particleCount, 512));
    const state = new PeribodyState(bodyId, cx, cy, cz, radius, mass, particleCount, rng,
      this.horizonFactor, this.E, this.sc, this.zeta);
    this._bodies.set(bodyId, state);
    return state;
  }

  removeBody(bodyId) {
    this._bodies.delete(bodyId);
  }

  hasBody(bodyId) { return this._bodies.has(bodyId); }

  /**
   * Update body center position (e.g. after rigid-body integration).
   * Sub-particles follow with rigid translation; internal strains are recomputed.
   */
  syncBodyPosition(bodyId, cx, cy, cz) {
    const state = this._bodies.get(bodyId);
    if (!state) return;
    const dx = cx - state.cx, dy = cy - state.cy, dz = cz - state.cz;
    state.cx = cx; state.cy = cy; state.cz = cz;
    const N = state.particleCount;
    const p = state.pos;
    for (let i = 0; i < N; i++) {
      p[i * 3]     += dx;
      p[i * 3 + 1] += dy;
      p[i * 3 + 2] += dz;
    }
  }

  /**
   * Apply an impact force at a world position, distributing over nearby particles.
   * impactForce: [fx, fy, fz] total impulse.
   */
  applyImpact(bodyId, wx, wy, wz, fx, fy, fz, radius) {
    const state = this._bodies.get(bodyId);
    if (!state || state.status === BODY_SHATTERED) return;

    const N = state.particleCount;
    const p = state.pos;
    const v = state.vel;
    const invMass = 1 / state.particleMass;
    const r2 = radius * radius;

    for (let i = 0; i < N; i++) {
      const dx = p[i * 3] - wx, dy = p[i * 3 + 1] - wy, dz = p[i * 3 + 2] - wz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < r2) {
        const weight = 1 - Math.sqrt(d2) / radius;
        v[i * 3]     += fx * weight * invMass;
        v[i * 3 + 1] += fy * weight * invMass;
        v[i * 3 + 2] += fz * weight * invMass;
      }
    }
  }

  /**
   * Apply tidal stress field: stretch bonds along a tidal axis.
   * tidalAcc: tidal acceleration magnitude.
   */
  applyTidalStress(bodyId, tidalAxis, tidalAcc) {
    const state = this._bodies.get(bodyId);
    if (!state || state.status === BODY_SHATTERED) return;

    const N = state.particleCount;
    const p = state.pos;
    const v = state.vel;
    const ax = tidalAxis[0] * tidalAcc;
    const ay = tidalAxis[1] * tidalAcc;
    const az = tidalAxis[2] * tidalAcc;

    for (let i = 0; i < N; i++) {
      // Sign based on position relative to center
      const relX = p[i * 3] - state.cx;
      const relY = p[i * 3 + 1] - state.cy;
      const relZ = p[i * 3 + 2] - state.cz;
      const sign = (relX * ax + relY * ay + relZ * az) > 0 ? 1 : -1;
      v[i * 3]     += sign * ax * 0.001;
      v[i * 3 + 1] += sign * ay * 0.001;
      v[i * 3 + 2] += sign * az * 0.001;
    }
  }

  /**
   * Step peridynamics for all registered bodies.
   * @param {number} dt  physics timestep
   * @returns {Array} fracture events (bodyId → fragments)
   */
  step(dt) {
    this._fractureEvents.length = 0;
    for (const [bodyId, state] of this._bodies) {
      if (state.status === BODY_SHATTERED) continue;
      this._stepBody(state, dt);
    }
    const events = [...this._fractureEvents];
    this._fractureEvents.length = 0;
    return events;
  }

  _stepBody(state, dt) {
    const N = state.particleCount;
    const p = state.pos;
    const v = state.vel;
    const f = state.force;

    // Zero forces
    f.fill(0);

    // Bond forces
    let bondsBrokenThisStep = 0;
    const bonds = state.bonds;
    const L0arr = state.bondL0;
    const broken = state.bondBroken;
    const stiffness = state.stiffness;
    const sc = state.sc;
    const damping = state.zeta;
    const nBonds = state.bondCount;

    for (let b = 0; b < nBonds; b++) {
      if (broken[b]) continue;

      const bi = bonds[b * 2];
      const bj = bonds[b * 2 + 1];
      const L0 = L0arr[b];

      const dx = p[bj * 3] - p[bi * 3];
      const dy = p[bj * 3 + 1] - p[bi * 3 + 1];
      const dz = p[bj * 3 + 2] - p[bi * 3 + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < 1e-12) continue;

      const stretch = (dist - L0) / L0;

      if (Math.abs(stretch) > sc) {
        broken[b] = 1;
        state.brokenCount++;
        bondsBrokenThisStep++;
        continue;
      }

      // Elastic force along bond
      const forceMag = stiffness * stretch;

      // Relative velocity for damping
      const dvx = v[bj * 3] - v[bi * 3];
      const dvy = v[bj * 3 + 1] - v[bi * 3 + 1];
      const dvz = v[bj * 3 + 2] - v[bi * 3 + 2];
      const dv_along = (dvx * dx + dvy * dy + dvz * dz) / dist;
      const dampMag = damping * dv_along;

      const total = (forceMag + dampMag) / dist;
      const fx = total * dx, fy = total * dy, fz = total * dz;

      f[bi * 3]     += fx; f[bi * 3 + 1] += fy; f[bi * 3 + 2] += fz;
      f[bj * 3]     -= fx; f[bj * 3 + 1] -= fy; f[bj * 3 + 2] -= fz;
    }

    // Integrate sub-particles (semi-implicit Euler)
    const invM = 1 / state.particleMass;
    for (let i = 0; i < N; i++) {
      v[i * 3]     += f[i * 3]     * invM * dt;
      v[i * 3 + 1] += f[i * 3 + 1] * invM * dt;
      v[i * 3 + 2] += f[i * 3 + 2] * invM * dt;

      p[i * 3]     += v[i * 3]     * dt;
      p[i * 3 + 1] += v[i * 3 + 1] * dt;
      p[i * 3 + 2] += v[i * 3 + 2] * dt;
    }

    // Recompute center
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < N; i++) {
      cx += p[i * 3]; cy += p[i * 3 + 1]; cz += p[i * 3 + 2];
    }
    state.cx = cx / N; state.cy = cy / N; state.cz = cz / N;

    // Update damage metric
    state.damage = state.brokenCount / Math.max(1, nBonds);

    if (state.damage > 0.3) state.status = BODY_DAMAGED;

    // Trigger shatter event if too damaged
    if (bondsBrokenThisStep > 0 && state.damage > 0.55) {
      this._triggerShatter(state);
    }
  }

  _triggerShatter(state) {
    if (state.status === BODY_SHATTERED) return;
    state.status = BODY_SHATTERED;

    // Find connected components of unbroken bonds
    const N = state.particleCount;
    const component = new Int32Array(N).fill(-1);
    let numComponents = 0;

    const adj = Array.from({ length: N }, () => []);
    for (let b = 0; b < state.bondCount; b++) {
      if (state.bondBroken[b]) continue;
      const bi = state.bonds[b * 2];
      const bj = state.bonds[b * 2 + 1];
      adj[bi].push(bj);
      adj[bj].push(bi);
    }

    // BFS to find components
    const queue = [];
    for (let start = 0; start < N; start++) {
      if (component[start] !== -1) continue;
      const comp = numComponents++;
      queue.length = 0;
      queue.push(start);
      component[start] = comp;
      let head = 0;
      while (head < queue.length) {
        const cur = queue[head++];
        for (const nb of adj[cur]) {
          if (component[nb] === -1) {
            component[nb] = comp;
            queue.push(nb);
          }
        }
      }
    }

    // Build fragment groups
    const fragments = Array.from({ length: numComponents }, () => ({
      indices: [], cx: 0, cy: 0, cz: 0,
      vx: 0, vy: 0, vz: 0, mass: 0,
    }));
    const p = state.pos, v = state.vel;
    for (let i = 0; i < N; i++) {
      const c = component[i];
      const frag = fragments[c];
      frag.indices.push(i);
      frag.cx += p[i * 3]; frag.cy += p[i * 3 + 1]; frag.cz += p[i * 3 + 2];
      frag.vx += v[i * 3]; frag.vy += v[i * 3 + 1]; frag.vz += v[i * 3 + 2];
      frag.mass += state.particleMass;
    }
    for (const frag of fragments) {
      const n = frag.indices.length;
      if (n === 0) continue;
      frag.cx /= n; frag.cy /= n; frag.cz /= n;
      frag.vx /= n; frag.vy /= n; frag.vz /= n;
      // Fragment radius from volume
      frag.radius = state.radius * Math.cbrt(n / state.particleCount);
    }

    this._fractureEvents.push({
      bodyId: state.bodyId,
      parentMass: state.mass,
      parentRadius: state.radius,
      fragments: fragments.filter(f => f.indices.length >= 2),
      singletons: fragments.filter(f => f.indices.length < 2).map(f => ({
        cx: f.cx, cy: f.cy, cz: f.cz,
        vx: f.vx, vy: f.vy, vz: f.vz, mass: f.mass, radius: f.radius * 0.5,
      })),
    });
  }

  /**
   * Get bond visualization data for debug overlay.
   * @returns {{active: Float32Array, broken: Float32Array}}
   */
  getBondLines(bodyId) {
    const state = this._bodies.get(bodyId);
    if (!state) return null;
    const active = [], broken = [];
    const p = state.pos;
    for (let b = 0; b < state.bondCount; b++) {
      const bi = state.bonds[b * 2];
      const bj = state.bonds[b * 2 + 1];
      const arr = state.bondBroken[b] ? broken : active;
      arr.push(p[bi * 3], p[bi * 3 + 1], p[bi * 3 + 2]);
      arr.push(p[bj * 3], p[bj * 3 + 1], p[bj * 3 + 2]);
    }
    return { active: new Float32Array(active), broken: new Float32Array(broken) };
  }

  getDamage(bodyId) {
    const state = this._bodies.get(bodyId);
    return state ? state.damage : 0;
  }

  getStatus(bodyId) {
    const state = this._bodies.get(bodyId);
    return state ? state.status : -1;
  }

  updateParams(E, sc, zeta) {
    this.E = E; this.sc = sc; this.zeta = zeta;
    for (const state of this._bodies.values()) {
      state.sc = sc; state.zeta = zeta;
      // Recompute stiffness from E
      state.stiffness = E * state.particleMass / state.particleCount;
    }
  }
}

// ─── PeribodyState ─────────────────────────────────────────────────────────────

class PeribodyState {
  constructor(bodyId, cx, cy, cz, radius, mass, particleCount, rng, horizonFactor, E, sc, zeta) {
    this.bodyId = bodyId;
    this.cx = cx; this.cy = cy; this.cz = cz;
    this.radius = radius;
    this.mass = mass;
    this.particleCount = particleCount;
    this.particleMass = mass / particleCount;

    this.sc = sc;
    this.zeta = zeta;
    this.stiffness = E * this.particleMass / particleCount;

    this.status = BODY_INTACT;
    this.damage = 0;
    this.brokenCount = 0;

    // Sub-particle arrays
    this.pos = new Float64Array(particleCount * 3);
    this.vel = new Float64Array(particleCount * 3);
    this.force = new Float64Array(particleCount * 3);

    // Populate particles in sphere
    const horizon = radius * horizonFactor;
    this._generateParticles(cx, cy, cz, radius, rng);

    // Build bond graph
    const { bonds, L0, count } = this._buildBonds(horizon);
    this.bonds = bonds;
    this.bondL0 = L0;
    this.bondBroken = new Uint8Array(count);
    this.bondCount = count;
  }

  _generateParticles(cx, cy, cz, radius, rng) {
    const N = this.particleCount;
    const p = this.pos;
    // Shell-core distribution: 70% inside sphere, 30% on surface shell
    const shellCount = Math.floor(N * 0.3);
    const coreCount = N - shellCount;

    for (let i = 0; i < coreCount; i++) {
      const [dx, dy, dz] = rng.inSphere(radius * 0.85);
      p[i * 3] = cx + dx; p[i * 3 + 1] = cy + dy; p[i * 3 + 2] = cz + dz;
    }
    for (let i = coreCount; i < N; i++) {
      const [ux, uy, uz] = rng.onSphere();
      const r = radius * rng.float(0.85, 1.0);
      p[i * 3] = cx + ux * r; p[i * 3 + 1] = cy + uy * r; p[i * 3 + 2] = cz + uz * r;
    }
  }

  _buildBonds(horizon) {
    const N = this.particleCount;
    const p = this.pos;
    const h2 = horizon * horizon;

    const bondList = [];
    const L0List = [];

    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = p[j * 3] - p[i * 3];
        const dy = p[j * 3 + 1] - p[i * 3 + 1];
        const dz = p[j * 3 + 2] - p[i * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < h2) {
          bondList.push(i, j);
          L0List.push(Math.sqrt(d2));
        }
      }
    }

    const count = bondList.length / 2;
    return {
      bonds: new Int32Array(bondList),
      L0: new Float64Array(L0List),
      count,
    };
  }
}
