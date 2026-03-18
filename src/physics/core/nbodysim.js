/**
 * N-Body Simulation Orchestrator (CPU path, runs inside physics.worker.js).
 *
 * Implements:
 *  - Kick-Drift-Kick (velocity Verlet / leapfrog) integrator
 *  - Adaptive CFL time step
 *  - Barnes–Hut gravity (O(N log N))
 *  - BVH broad-phase + analytic narrow-phase collision detection
 *  - Baumgarte positional correction for overlapping bodies
 *  - Peridynamics fracture
 *  - Voronoi shatter on heavy impact
 *  - SPH ejecta
 *  - Tidal disruption / Roche limit check
 *  - Energy monitoring
 */

import { BarnesHut }         from '../cpu/barneshut.js';
import { BVH }               from '../cpu/bvh.js';
import { PeridynamicsSystem } from '../fracture/peridynamics.js';
import { VoronoiShatter }    from '../fracture/voronoi.js';
import { SPHSystem }         from '../fluids/sph.js';
import { RNG }               from '../../util/rng.js';

// Body types
export const TYPE_STAR     = 0;
export const TYPE_PLANET   = 1;
export const TYPE_RUBBLE   = 2;
export const TYPE_FRAGMENT = 3;

// Max bodies budget
export const MAX_BODIES = 512;
export const MAX_FRAGMENTS = 256;

export class NBodySim {
  constructor(config = {}) {
    // Simulation parameters
    this.G          = config.G          ?? 1.0;
    this.theta      = config.theta      ?? 0.6;
    this.epsilon    = config.epsilon    ?? 0.03;
    this.dtMin      = config.dtMin      ?? 1e-5;
    this.dtMax      = config.dtMax      ?? 0.02;
    this.cfl        = config.cfl        ?? 0.4;
    this.cMinSpeed  = config.cMinSpeed  ?? 0.1;
    this.restitution = config.restitution ?? 0.3;
    this.baumgarte  = config.baumgarte  ?? 0.2;

    // Peridynamics params
    this.pdE        = config.pdE        ?? 5e3;
    this.pdSc       = config.pdSc       ?? 0.25;
    this.pdZeta     = config.pdZeta     ?? 0.05;
    this.pdParticles = config.pdParticles ?? 64;

    // Cohesion/Roche
    this.cohesion   = config.cohesion   ?? 0.5;

    // SPH
    this.sphH       = config.sphH       ?? 1.5;
    this.sphK       = config.sphK       ?? 2.0;

    // RNG
    this._rng = new RNG(config.seed ?? 42);

    // Body arrays (flat, stride-3 for vec3)
    this._N = 0;
    this.pos  = new Float64Array(MAX_BODIES * 3);
    this.vel  = new Float64Array(MAX_BODIES * 3);
    this.acc  = new Float64Array(MAX_BODIES * 3);
    this.mass = new Float64Array(MAX_BODIES);
    this.radii= new Float64Array(MAX_BODIES);
    this.types= new Uint8Array(MAX_BODIES);
    this.alive= new Uint8Array(MAX_BODIES);
    this.stress = new Float32Array(MAX_BODIES);
    this.color = new Float32Array(MAX_BODIES * 3);

    // Sub-systems
    this._bh     = new BarnesHut(this.theta, this.epsilon, this.G);
    this._bvh    = new BVH();
    this._peri   = new PeridynamicsSystem({ E: this.pdE, sc: this.pdSc, zeta: this.pdZeta });
    this._voron  = new VoronoiShatter();
    this._sph    = new SPHSystem({ h: this.sphH, k: this.sphK });

    // Energy tracking
    this._energy0 = null;
    this.energyError = 0;

    // Simulation time
    this.time = 0;
    this.dt   = this.dtMax;
    this.step = 0;

    // Pending new-body events for renderer
    this._newBodies = [];
    this._removedIds = [];

    // Contact history for warm start (optional, kept for Baumgarte)
    this._contacts = [];

    // Active index scratch (populated by _getActiveN)
    this._activeIdx = [];
  }

  // ─── Initialization ───────────────────────────────────────────────────────

  reset(scenario, seed) {
    this._rng.reset(seed ?? this._rng.seed);
    this._N = 0;
    this.pos.fill(0); this.vel.fill(0); this.acc.fill(0);
    this.mass.fill(0); this.radii.fill(0); this.alive.fill(0);
    this.stress.fill(0);
    this._peri = new PeridynamicsSystem({ E: this.pdE, sc: this.pdSc, zeta: this.pdZeta });
    this._sph.clear();
    this.time = 0; this.step = 0;
    this._energy0 = null;
    this._newBodies.length = 0;
    this._removedIds.length = 0;

    if (scenario) this.loadScenario(scenario);
    return this;
  }

  loadScenario(name) {
    const rng = this._rng;
    switch (name) {
      case 'headon':    this._scenarioHeadOn(rng);    break;
      case 'grazing':   this._scenarioGrazing(rng);   break;
      case 'roche':     this._scenarioRoche(rng);     break;
      case 'tribody':   this._scenarioTriBody(rng);   break;
      case 'ring':      this._scenarioDebrisRing(rng); break;
      default:          this._scenarioHeadOn(rng);
    }
    this._computeInitialEnergy();
  }

  /** Add a single body; returns its index. */
  addBody(px, py, pz, vx, vy, vz, mass, radius, type, color) {
    if (this._N >= MAX_BODIES) return -1;
    const i = this._N++;
    this.pos[i*3]=px; this.pos[i*3+1]=py; this.pos[i*3+2]=pz;
    this.vel[i*3]=vx; this.vel[i*3+1]=vy; this.vel[i*3+2]=vz;
    this.mass[i] = mass;
    this.radii[i] = radius;
    this.types[i] = type ?? TYPE_PLANET;
    this.alive[i] = 1;
    this.color[i*3]   = color?.[0] ?? 0.6;
    this.color[i*3+1] = color?.[1] ?? 0.6;
    this.color[i*3+2] = color?.[2] ?? 0.6;
    return i;
  }

  // ─── Scenarios ────────────────────────────────────────────────────────────

  _scenarioHeadOn(rng) {
    // Two large bodies on collision course
    const r1 = 2.5, r2 = 2.0;
    const m1 = 8.0, m2 = 5.0;
    this.addBody(-12, rng.float(-0.5, 0.5), 0,  2.5, 0, 0, m1, r1, TYPE_PLANET, [0.3,0.5,0.9]);
    this.addBody( 12, rng.float(-0.5, 0.5), 0, -2.5, 0, 0, m2, r2, TYPE_PLANET, [0.8,0.4,0.2]);
    // Register for peridynamics
    this._peri.registerBody(0, -12, 0, 0, r1, m1, this.pdParticles, rng);
    this._peri.registerBody(1,  12, 0, 0, r2, m2, this.pdParticles, rng);
  }

  _scenarioGrazing(rng) {
    const r1 = 3.0, r2 = 2.2;
    const m1 = 10.0, m2 = 6.0;
    // Impact parameter = combined radii * 0.85
    const b = (r1 + r2) * 0.85;
    this.addBody(-14, b, 0,  2.8, 0, 0, m1, r1, TYPE_PLANET, [0.2,0.7,0.5]);
    this.addBody( 14, 0, 0, -2.8, 0, 0, m2, r2, TYPE_PLANET, [0.9,0.6,0.2]);
    this._peri.registerBody(0, -14, b, 0, r1, m1, this.pdParticles, rng);
    this._peri.registerBody(1,  14, 0, 0, r2, m2, this.pdParticles, rng);
  }

  _scenarioRoche(rng) {
    // Large primary + smaller body skimming within Roche limit
    const M = 20.0, R = 5.0; // primary
    const m = 1.0, r = 1.0;  // satellite
    // Roche limit ≈ 2.44 * R * (M/m)^(1/3)
    const roche = 2.44 * R * Math.cbrt(M / m);
    // Orbit just inside Roche limit
    const orbitR = roche * 0.92;
    const vCirc = Math.sqrt(this.G * M / orbitR);
    this.addBody(0, 0, 0, 0, 0, 0, M, R, TYPE_STAR, [1.0, 0.9, 0.5]);
    this.addBody(orbitR, 0, 0, 0, vCirc * 0.85, 0, m, r, TYPE_PLANET, [0.4, 0.6, 0.9]);
    this._peri.registerBody(1, orbitR, 0, 0, r, m, this.pdParticles, rng);
  }

  _scenarioTriBody(rng) {
    // Stable figure-8 three-body starting configuration (Chenciner-Montgomery)
    const m = 5.0;
    const scale = 8.0, vScale = 1.0;
    // Approximate figure-8 initial conditions
    const positions = [
      [-0.97000436, 0.24308753, 0],
      [ 0,          0,          0],
      [ 0.97000436,-0.24308753, 0],
    ];
    const velocities = [
      [0.93240737/2,  0.86473146/2, 0],
      [-0.93240737,  -0.86473146,   0],
      [0.93240737/2,  0.86473146/2, 0],
    ];
    const colors = [[0.9,0.4,0.2],[0.2,0.7,0.9],[0.6,0.9,0.3]];
    const r = 1.8;
    for (let k = 0; k < 3; k++) {
      const [px,py,pz] = positions[k].map(v => v * scale);
      const [vx,vy,vz] = velocities[k].map(v => v * vScale);
      const i = this.addBody(px, py, pz, vx, vy, vz, m, r, TYPE_PLANET, colors[k]);
      this._peri.registerBody(i, px, py, pz, r, m, this.pdParticles, rng);
    }
  }

  _scenarioDebrisRing(rng) {
    // Central massive body + ring of small bodies
    const M = 15.0, R = 4.0;
    this.addBody(0, 0, 0, 0, 0, 0, M, R, TYPE_STAR, [1.0, 0.85, 0.4]);

    const ringCount = Math.min(40, MAX_BODIES - 1);
    const ringR = 12, spread = 2.5;
    for (let k = 0; k < ringCount; k++) {
      const angle = (k / ringCount) * Math.PI * 2;
      const orbitR = ringR + rng.float(-spread, spread);
      const vCirc = Math.sqrt(this.G * M / orbitR);
      const px = Math.cos(angle) * orbitR;
      const py = rng.float(-0.5, 0.5);
      const pz = Math.sin(angle) * orbitR;
      const vx = -Math.sin(angle) * vCirc;
      const vy = rng.float(-0.1, 0.1);
      const vz =  Math.cos(angle) * vCirc;
      const r = rng.float(0.2, 0.6);
      const m = r * r * r * 0.3;
      const c = [rng.float(0.4,0.9), rng.float(0.4,0.9), rng.float(0.4,0.9)];
      this.addBody(px, py, pz, vx, vy, vz, m, r, TYPE_RUBBLE, c);
    }
  }

  // ─── Main Step ────────────────────────────────────────────────────────────

  /**
   * Advance simulation by one leapfrog step.
   * Returns performance metrics.
   */
  stepPhysics() {
    const t0 = performance.now();

    const N = this._activeCount();
    if (N === 0) return { physicsMs: 0, fracEvents: [] };

    // Adaptive time step
    this.dt = this._computeDt();

    // --- Kick-Drift-Kick leapfrog ---

    // Half-kick: v += 0.5 * dt * a
    this._halfKick(this.dt);

    // Drift: x += dt * v
    this._drift(this.dt);

    // Sync peridynamics positions
    this._syncPeridynamicsPositions();

    // Rebuild gravity tree
    this._bh.theta = this.theta;
    this._bh.epsilon2 = this.epsilon * this.epsilon;
    const activePos = this._getActivePos();
    const activeMass = this._getActiveMass();
    const activeN = this._getActiveN();

    this._bh.buildTree(activePos, activeMass, activeN);
    this._bh.setLeafMasses(activePos, activeMass, activeN);

    // Compute gravity accelerations
    const activeAcc = new Float64Array(activeN * 3);
    this._bh.computeAccelerations(activePos, activeMass, activeAcc, activeN);
    this._scatterAcc(activeAcc);

    // Add tidal forces
    this._computeTidalForces();

    // Add peridynamics forces
    this._addPeridynamicsForces();

    // Second half-kick
    this._halfKick(this.dt);

    // Broad-phase collision detection
    const allRadii = this._getActiveRadii();
    this._bvh.build(activePos, allRadii, activeN);
    const pairs = this._bvh.queryPairs();

    // Narrow-phase + resolve
    const fracEvents = [];
    for (const [ai, aj] of pairs) {
      const i = this._activeIdx[ai];
      const j = this._activeIdx[aj];
      const event = this._resolveCollision(i, j);
      if (event) fracEvents.push(event);
    }

    // Process fracture events
    const pdEvents = this._peri.step(this.dt);
    for (const ev of pdEvents) fracEvents.push(ev);

    // Despawn shattered primary bodies
    this._processFractureEvents(fracEvents, this._rng);

    // Step SPH
    this._sph.step(this.dt, (i, pos, out) => {
      // Simple gravity toward nearest heavy body
      this._sphExternalGravity(i, pos, out);
    });

    // Energy monitoring
    if (this.step % 100 === 0) this._updateEnergyError(activePos, activeMass, activeN);

    this.time += this.dt;
    this.step++;

    return {
      physicsMs: performance.now() - t0,
      fracEvents,
    };
  }

  // ─── Integrator ───────────────────────────────────────────────────────────

  _halfKick(dt) {
    const half = dt * 0.5;
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      this.vel[i*3]   += this.acc[i*3]   * half;
      this.vel[i*3+1] += this.acc[i*3+1] * half;
      this.vel[i*3+2] += this.acc[i*3+2] * half;
    }
  }

  _drift(dt) {
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      this.pos[i*3]   += this.vel[i*3]   * dt;
      this.pos[i*3+1] += this.vel[i*3+1] * dt;
      this.pos[i*3+2] += this.vel[i*3+2] * dt;
    }
  }

  _computeDt() {
    let minH = Infinity;
    const cMin = this.cMinSpeed;
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      const vx = this.vel[i*3], vy = this.vel[i*3+1], vz = this.vel[i*3+2];
      const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
      const h = this.radii[i] / Math.max(speed, cMin);
      if (h < minH) minH = h;
    }
    const dt = this.cfl * minH;
    return Math.max(this.dtMin, Math.min(this.dtMax, dt));
  }

  // ─── Active body helpers ──────────────────────────────────────────────────

  _getActiveN() {
    this._activeIdx.length = 0;
    for (let i = 0; i < MAX_BODIES; i++) {
      if (this.alive[i]) this._activeIdx.push(i);
    }
    return this._activeIdx.length;
  }

  _activeCount() {
    let c = 0;
    for (let i = 0; i < MAX_BODIES; i++) if (this.alive[i]) c++;
    return c;
  }

  _getActivePos() {
    const N = this._activeIdx.length;
    const out = new Float64Array(N * 3);
    for (let ai = 0; ai < N; ai++) {
      const i = this._activeIdx[ai];
      out[ai*3]=this.pos[i*3]; out[ai*3+1]=this.pos[i*3+1]; out[ai*3+2]=this.pos[i*3+2];
    }
    return out;
  }

  _getActiveMass() {
    const N = this._activeIdx.length;
    const out = new Float64Array(N);
    for (let ai = 0; ai < N; ai++) out[ai] = this.mass[this._activeIdx[ai]];
    return out;
  }

  _getActiveRadii() {
    const N = this._activeIdx.length;
    const out = new Float64Array(N);
    for (let ai = 0; ai < N; ai++) out[ai] = this.radii[this._activeIdx[ai]];
    return out;
  }

  _scatterAcc(activeAcc) {
    const N = this._activeIdx.length;
    for (let ai = 0; ai < N; ai++) {
      const i = this._activeIdx[ai];
      this.acc[i*3]   = activeAcc[ai*3];
      this.acc[i*3+1] = activeAcc[ai*3+1];
      this.acc[i*3+2] = activeAcc[ai*3+2];
    }
  }

  // ─── Tidal Forces ─────────────────────────────────────────────────────────

  _computeTidalForces() {
    const G = this.G;
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      if (this.types[i] === TYPE_STAR) continue;

      let maxTidal = 0, tidAxis = [1, 0, 0];

      for (let j = 0; j < MAX_BODIES; j++) {
        if (!this.alive[j] || i === j) continue;
        const dx = this.pos[j*3]-this.pos[i*3];
        const dy = this.pos[j*3+1]-this.pos[i*3+1];
        const dz = this.pos[j*3+2]-this.pos[i*3+2];
        const r2 = dx*dx+dy*dy+dz*dz;
        const r  = Math.sqrt(r2);
        const r3 = r2 * r;

        // Tidal stress ~ 2 G M R_i / r^3 * cohesionFactor
        const sigma = 2 * G * this.mass[j] * this.radii[i] / (r3 * this.cohesion);
        if (sigma > maxTidal) {
          maxTidal = sigma;
          const il = 1 / r;
          tidAxis = [dx*il, dy*il, dz*il];
        }
      }

      this.stress[i] = Math.min(1, maxTidal);

      // Apply tidal to peridynamics body if registered
      if (maxTidal > 0.1 && this._peri.hasBody(i)) {
        this._peri.applyTidalStress(i, tidAxis, maxTidal * 0.01);
      }
    }
  }

  _addPeridynamicsForces() {
    // No-op for rigid bodies; peridynamics is internal only.
    // If needed, add net force from peridynamics to rigid body here.
  }

  _syncPeridynamicsPositions() {
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      if (this._peri.hasBody(i)) {
        this._peri.syncBodyPosition(i, this.pos[i*3], this.pos[i*3+1], this.pos[i*3+2]);
      }
    }
  }

  // ─── Collision Resolution ─────────────────────────────────────────────────

  _resolveCollision(i, j) {
    if (!this.alive[i] || !this.alive[j]) return null;

    const dx = this.pos[j*3]-this.pos[i*3];
    const dy = this.pos[j*3+1]-this.pos[i*3+1];
    const dz = this.pos[j*3+2]-this.pos[i*3+2];
    const dist2 = dx*dx+dy*dy+dz*dz;
    const sumR = this.radii[i] + this.radii[j];

    if (dist2 >= sumR * sumR) return null;

    const dist = Math.sqrt(dist2) || 1e-10;
    const nx = dx/dist, ny = dy/dist, nz = dz/dist;

    // Relative velocity
    const rvx = this.vel[j*3]-this.vel[i*3];
    const rvy = this.vel[j*3+1]-this.vel[i*3+1];
    const rvz = this.vel[j*3+2]-this.vel[i*3+2];
    const rvN = rvx*nx + rvy*ny + rvz*nz;

    if (rvN > 0) return null; // separating

    // Impulse scalar
    const mi = this.mass[i], mj = this.mass[j];
    const invMi = 1/mi, invMj = 1/mj;
    const e = this.restitution;
    const j_imp = -(1 + e) * rvN / (invMi + invMj);

    this.vel[i*3]   -= j_imp * invMi * nx;
    this.vel[i*3+1] -= j_imp * invMi * ny;
    this.vel[i*3+2] -= j_imp * invMi * nz;
    this.vel[j*3]   += j_imp * invMj * nx;
    this.vel[j*3+1] += j_imp * invMj * ny;
    this.vel[j*3+2] += j_imp * invMj * nz;

    // Baumgarte positional correction
    const penetration = sumR - dist;
    const corr = (penetration / (invMi + invMj)) * this.baumgarte;
    this.pos[i*3]   -= corr * invMi * nx;
    this.pos[i*3+1] -= corr * invMi * ny;
    this.pos[i*3+2] -= corr * invMi * nz;
    this.pos[j*3]   += corr * invMj * nx;
    this.pos[j*3+1] += corr * invMj * ny;
    this.pos[j*3+2] += corr * invMj * nz;

    // Impact speed
    const impactSpeed = Math.abs(j_imp) * (invMi + invMj);

    // Update stress
    const stressVal = Math.min(1, impactSpeed / 5);
    this.stress[i] = Math.max(this.stress[i], stressVal);
    this.stress[j] = Math.max(this.stress[j], stressVal);

    // Apply impact to peridynamics
    const impX = this.pos[i*3] + nx * this.radii[i];
    const impY = this.pos[i*3+1] + ny * this.radii[i];
    const impZ = this.pos[i*3+2] + nz * this.radii[i];

    const fMag = j_imp;
    if (this._peri.hasBody(i)) {
      this._peri.applyImpact(i, impX, impY, impZ,
        -nx*fMag*invMi, -ny*fMag*invMi, -nz*fMag*invMi,
        this.radii[i] * 0.4);
    }
    if (this._peri.hasBody(j)) {
      this._peri.applyImpact(j, impX, impY, impZ,
        nx*fMag*invMj, ny*fMag*invMj, nz*fMag*invMj,
        this.radii[j] * 0.4);
    }

    // Spawn SPH ejecta on significant impact
    if (impactSpeed > 0.5) {
      const ejectCount = Math.min(12, (impactSpeed * 4) | 0);
      const impactPoint = {
        x: impX, y: impY, z: impZ,
        normalX: nx, normalY: ny, normalZ: nz,
      };
      const ejecta1 = this._voron.generateEjecta(
        { cx: this.pos[i*3], cy: this.pos[i*3+1], cz: this.pos[i*3+2],
          radius: this.radii[i], mass: mi,
          vel: [this.vel[i*3], this.vel[i*3+1], this.vel[i*3+2]] },
        impactPoint, impactSpeed, ejectCount, this._rng
      );
      this._sph.spawnEjecta(ejecta1);

      // Trigger Voronoi shatter if impact is very heavy
      if (impactSpeed > 2.0 && this._N < MAX_BODIES + MAX_FRAGMENTS) {
        const smallerBody = mi < mj ? i : j;
        const numFrag = Math.min(12, (impactSpeed * 2) | 0);
        return {
          type: 'voronoi_shatter',
          bodyIndex: smallerBody,
          impactPoint,
          numFragments: numFrag,
          impactSpeed,
        };
      }
    }

    return null;
  }

  // ─── Fracture Processing ───────────────────────────────────────────────────

  _processFractureEvents(events, rng) {
    for (const ev of events) {
      if (!ev) continue;

      if (ev.type === 'voronoi_shatter') {
        this._doVoronoiShatter(ev.bodyIndex, ev.impactPoint, ev.numFragments, rng);
      } else if (ev.bodyId !== undefined && ev.fragments) {
        // Peridynamics shatter event
        this._doPeridynamicsShatter(ev, rng);
      }
    }
  }

  _doVoronoiShatter(bodyIdx, impactPoint, numFrag, rng) {
    if (!this.alive[bodyIdx]) return;
    if (this._N >= MAX_BODIES) return;

    const body = {
      cx: this.pos[bodyIdx*3], cy: this.pos[bodyIdx*3+1], cz: this.pos[bodyIdx*3+2],
      radius: this.radii[bodyIdx],
      mass: this.mass[bodyIdx],
      vel: [this.vel[bodyIdx*3], this.vel[bodyIdx*3+1], this.vel[bodyIdx*3+2]],
    };

    // Only shatter planets/rubble, not stars
    if (this.types[bodyIdx] === TYPE_STAR) return;

    const fragments = this._voron.shatter(body, impactPoint, numFrag, rng);

    // Remove original
    this.alive[bodyIdx] = 0;
    this._peri.removeBody(bodyIdx);
    this._removedIds.push(bodyIdx);

    // Add fragments
    for (const frag of fragments) {
      const newIdx = this.addBody(
        frag.cx, frag.cy, frag.cz,
        frag.vx, frag.vy, frag.vz,
        frag.mass, frag.radius, TYPE_FRAGMENT,
        [this.color[bodyIdx*3], this.color[bodyIdx*3+1], this.color[bodyIdx*3+2]]
      );
      if (newIdx < 0) break;
      this.stress[newIdx] = frag.stress;
    }
  }

  _doPeridynamicsShatter(ev, rng) {
    const bodyIdx = ev.bodyId;
    if (!this.alive[bodyIdx]) return;

    this.alive[bodyIdx] = 0;
    this._removedIds.push(bodyIdx);

    for (const frag of ev.fragments) {
      if (this._N >= MAX_BODIES) break;
      const newIdx = this.addBody(
        frag.cx, frag.cy, frag.cz,
        frag.vx, frag.vy, frag.vz,
        frag.mass, frag.radius ?? this.radii[bodyIdx] * 0.3,
        TYPE_FRAGMENT,
        [this.color[bodyIdx*3] * 0.9, this.color[bodyIdx*3+1] * 0.8, this.color[bodyIdx*3+2] * 0.7]
      );
      if (newIdx >= 0) this.stress[newIdx] = 0.6;
    }
  }

  // ─── SPH gravity ─────────────────────────────────────────────────────────

  _sphExternalGravity(sphIdx, pos, out) {
    let ax = 0, ay = 0, az = 0;
    const px = pos[sphIdx*3], py = pos[sphIdx*3+1], pz = pos[sphIdx*3+2];
    const G = this.G, eps2 = this.epsilon * this.epsilon;

    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      const dx = this.pos[i*3]-px, dy = this.pos[i*3+1]-py, dz = this.pos[i*3+2]-pz;
      const r2 = dx*dx+dy*dy+dz*dz+eps2;
      const r3 = r2 * Math.sqrt(r2);
      const f = G * this.mass[i] / r3;
      ax += f*dx; ay += f*dy; az += f*dz;
    }
    out[0] = ax; out[1] = ay; out[2] = az;
  }

  // ─── Energy ───────────────────────────────────────────────────────────────

  _computeInitialEnergy() {
    const N = this._getActiveN();
    const pos = this._getActivePos();
    const mass = this._getActiveMass();
    this._energy0 = this._computeTotalEnergy(pos, mass, N);
  }

  _updateEnergyError(activePos, activeMass, activeN) {
    if (!this._energy0 || Math.abs(this._energy0) < 1e-30) return;
    const E = this._computeTotalEnergy(activePos, activeMass, activeN);
    this.energyError = Math.abs((E - this._energy0) / this._energy0);
  }

  _computeTotalEnergy(pos, mass, N) {
    let KE = 0, PE = 0;
    for (let ai = 0; ai < N; ai++) {
      const i = this._activeIdx[ai] ?? ai;
      const vx = this.vel[i*3], vy = this.vel[i*3+1], vz = this.vel[i*3+2];
      KE += 0.5 * mass[ai] * (vx*vx + vy*vy + vz*vz);
      for (let aj = ai + 1; aj < N; aj++) {
        const dx = pos[aj*3]-pos[ai*3], dy = pos[aj*3+1]-pos[ai*3+1], dz = pos[aj*3+2]-pos[ai*3+2];
        const r = Math.sqrt(dx*dx+dy*dy+dz*dz+this.epsilon*this.epsilon);
        PE -= this.G * mass[ai] * mass[aj] / r;
      }
    }
    return KE + PE;
  }

  // ─── Impulse tool ─────────────────────────────────────────────────────────

  /** Slice a body with a plane, producing two fragments. */
  sliceBody(bodyIdx, planePoint, planeNormal) {
    if (!this.alive[bodyIdx]) return;
    if (this.types[bodyIdx] === TYPE_STAR) return;

    const body = {
      cx: this.pos[bodyIdx*3], cy: this.pos[bodyIdx*3+1], cz: this.pos[bodyIdx*3+2],
      radius: this.radii[bodyIdx],
      mass: this.mass[bodyIdx],
      vel: [this.vel[bodyIdx*3], this.vel[bodyIdx*3+1], this.vel[bodyIdx*3+2]],
    };

    const halves = this._voron.sliceBody(body, planePoint, planeNormal, this._rng);
    if (halves.length < 2) return;

    this.alive[bodyIdx] = 0;
    this._peri.removeBody(bodyIdx);
    this._removedIds.push(bodyIdx);

    for (const frag of halves) {
      const newIdx = this.addBody(
        frag.cx, frag.cy, frag.cz,
        frag.vx, frag.vy, frag.vz,
        frag.mass, frag.radius, TYPE_FRAGMENT,
        [this.color[bodyIdx*3], this.color[bodyIdx*3+1], this.color[bodyIdx*3+2]]
      );
      if (newIdx >= 0) this.stress[newIdx] = 0.4;
    }
  }

  applyImpulse(bodyIdx, ix, iy, iz) {
    if (!this.alive[bodyIdx]) return;
    const invM = 1 / this.mass[bodyIdx];
    this.vel[bodyIdx*3]   += ix * invM;
    this.vel[bodyIdx*3+1] += iy * invM;
    this.vel[bodyIdx*3+2] += iz * invM;
  }

  findNearestBody(wx, wy, wz) {
    let minDist = Infinity, nearest = -1;
    for (let i = 0; i < MAX_BODIES; i++) {
      if (!this.alive[i]) continue;
      const dx = this.pos[i*3]-wx, dy = this.pos[i*3+1]-wy, dz = this.pos[i*3+2]-wz;
      const d = dx*dx+dy*dy+dz*dz;
      if (d < minDist) { minDist = d; nearest = i; }
    }
    return nearest;
  }

  // ─── Snapshot for renderer ────────────────────────────────────────────────

  getSnapshot() {
    const N = this._activeCount();
    const alive = [];
    for (let i = 0; i < MAX_BODIES; i++) if (this.alive[i]) alive.push(i);

    const positions  = new Float32Array(N * 3);
    const velocities = new Float32Array(N * 3);
    const masses     = new Float32Array(N);
    const radii      = new Float32Array(N);
    const types      = new Uint8Array(N);
    const colors     = new Float32Array(N * 3);
    const stresses   = new Float32Array(N);
    const ids        = new Int32Array(N);

    for (let ai = 0; ai < N; ai++) {
      const i = alive[ai];
      positions[ai*3]  = this.pos[i*3];
      positions[ai*3+1]= this.pos[i*3+1];
      positions[ai*3+2]= this.pos[i*3+2];
      velocities[ai*3] = this.vel[i*3];
      velocities[ai*3+1]=this.vel[i*3+1];
      velocities[ai*3+2]=this.vel[i*3+2];
      masses[ai]  = this.mass[i];
      radii[ai]   = this.radii[i];
      types[ai]   = this.types[i];
      colors[ai*3]  = this.color[i*3];
      colors[ai*3+1]= this.color[i*3+1];
      colors[ai*3+2]= this.color[i*3+2];
      stresses[ai]= this.stress[i];
      ids[ai]     = i;
    }

    const sphSnap = this._sph.getSnapshot();

    // Bond lines for debug
    const bondData = {};
    for (const i of alive) {
      if (this._peri.hasBody(i)) {
        bondData[i] = this._peri.getBondLines(i);
      }
    }

    const removedIds = [...this._removedIds];
    this._removedIds.length = 0;

    return {
      N, positions, velocities, masses, radii, types, colors, stresses, ids,
      sph: sphSnap,
      bondData,
      time: this.time,
      dt: this.dt,
      energyError: this.energyError,
      removedIds,
      physicsMs: 0, // filled in by worker after stepPhysics()
    };
  }

  // ─── Config update ────────────────────────────────────────────────────────

  updateConfig(cfg) {
    if (cfg.theta    !== undefined) { this.theta    = cfg.theta;    this._bh.theta = cfg.theta; }
    if (cfg.epsilon  !== undefined) { this.epsilon  = cfg.epsilon;  this._bh.epsilon2 = cfg.epsilon * cfg.epsilon; }
    if (cfg.dtMin    !== undefined) this.dtMin    = cfg.dtMin;
    if (cfg.dtMax    !== undefined) this.dtMax    = cfg.dtMax;
    if (cfg.cfl      !== undefined) this.cfl      = cfg.cfl;
    if (cfg.restitution !== undefined) this.restitution = cfg.restitution;
    if (cfg.cohesion !== undefined) this.cohesion = cfg.cohesion;
    if (cfg.pdE      !== undefined || cfg.pdSc !== undefined || cfg.pdZeta !== undefined) {
      if (cfg.pdE)   this.pdE   = cfg.pdE;
      if (cfg.pdSc)  this.pdSc  = cfg.pdSc;
      if (cfg.pdZeta)this.pdZeta= cfg.pdZeta;
      this._peri.updateParams(this.pdE, this.pdSc, this.pdZeta);
    }
    if (cfg.sphH !== undefined || cfg.sphK !== undefined) {
      this._sph.updateParams(cfg.sphH ?? this.sphH, cfg.sphK ?? this.sphK, this._sph.gamma);
    }
  }

  // ─── Self-test ────────────────────────────────────────────────────────────

  /**
   * Two-body circular orbit energy drift test.
   * Returns { passed, driftPercent } after N steps.
   */
  selfTestOrbitDrift(steps = 10000) {
    const saved = { N: this._N, pos: this.pos.slice(), vel: this.vel.slice(),
                    mass: this.mass.slice(), alive: this.alive.slice(),
                    radii: this.radii.slice(), types: this.types.slice() };

    this._N = 0;
    this.pos.fill(0); this.vel.fill(0); this.alive.fill(0);

    const m1 = 10, m2 = 1, sep = 15;
    const vCirc = Math.sqrt(this.G * (m1 + m2) / sep);
    const mu = m1 * m2 / (m1 + m2);
    const r1 = sep * m2 / (m1 + m2);
    const r2 = sep * m1 / (m1 + m2);

    this.addBody(-r1, 0, 0,  0, -vCirc * m2/(m1+m2), 0, m1, 1, TYPE_PLANET, [1,1,1]);
    this.addBody( r2, 0, 0,  0,  vCirc * m1/(m1+m2), 0, m2, 0.5, TYPE_PLANET, [1,1,1]);
    this._computeInitialEnergy();

    for (let s = 0; s < steps; s++) this.stepPhysics();

    const drift = this.energyError * 100;

    // Restore
    this._N = saved.N;
    this.pos.set(saved.pos); this.vel.set(saved.vel);
    this.mass.set(saved.mass); this.alive.set(saved.alive);
    this.radii.set(saved.radii); this.types.set(saved.types);

    return { passed: drift < 0.5, driftPercent: drift };
  }
}
