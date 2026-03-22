/**
 * Lightweight SPH (Smoothed Particle Hydrodynamics) for ejecta simulation.
 *
 * Density:    ρ_i = Σ_j m_j W_poly6(r_ij, h)
 * Pressure:   p_i = k * ((ρ_i/ρ0)^γ - 1)   [Tait EOS, γ=7]
 * Force:      a_i = -Σ_j m_j (p_i/ρ_i² + p_j/ρ_j²) ∇W_spiky(r_ij, h)
 *                   + viscosity term (Monaghan)
 *                   + gravity from external field
 *
 * Particles fade (alpha → 0) over their lifetime and are despawned.
 */

export const MAX_SPH = 4096;

// Kernel constants (pre-multiplied)
const POLY6_COEF = 315 / (64 * Math.PI); // per h^9
const SPIKY_COEF = -45 / Math.PI;        // per h^6

export class SPHSystem {
  /**
   * @param {Object} opts
   * @param {number} opts.h          Smoothing radius
   * @param {number} opts.rho0       Rest density
   * @param {number} opts.k          Pressure stiffness
   * @param {number} opts.gamma      EOS exponent
   * @param {number} opts.viscosity  Artificial viscosity (Monaghan alpha)
   * @param {number} opts.gravity    External gravity magnitude (pointing down −Y)
   */
  constructor(opts = {}) {
    this.h       = opts.h        ?? 1.5;
    this.rho0    = opts.rho0     ?? 1.0;
    this.k       = opts.k        ?? 2.0;
    this.gamma   = opts.gamma    ?? 7;
    this.visc    = opts.viscosity ?? 0.08;
    this.gravity = opts.gravity  ?? 0;

    this.h2   = this.h * this.h;
    this.h6   = this.h2 * this.h2 * this.h2;
    this.h9   = this.h6 * this.h2 * this.h;
    this.h6s  = this.h6; // for spiky

    // Particle arrays
    this.count = 0;
    this.pos       = new Float32Array(MAX_SPH * 3);
    this.vel       = new Float32Array(MAX_SPH * 3);
    this.acc       = new Float32Array(MAX_SPH * 3);
    this.mass      = new Float32Array(MAX_SPH);
    this.density   = new Float32Array(MAX_SPH);
    this.pressure  = new Float32Array(MAX_SPH);
    this.lifetime  = new Float32Array(MAX_SPH);  // seconds remaining
    this.maxLife   = new Float32Array(MAX_SPH);
    this.temp      = new Float32Array(MAX_SPH);  // 0..1 for color
    this.active    = new Uint8Array(MAX_SPH);
  }

  // ─── Spawn ─────────────────────────────────────────────────────────────────

  /**
   * Spawn ejecta particles from an ejecta descriptor array.
   * @param {Array<{x,y,z,vx,vy,vz,mass,lifetime,temperature}>} ejecta
   */
  spawnEjecta(ejecta) {
    for (const p of ejecta) {
      if (this.count >= MAX_SPH) break;
      const i = this._findSlot();
      if (i < 0) break;
      this.pos[i * 3]     = p.x;
      this.pos[i * 3 + 1] = p.y;
      this.pos[i * 3 + 2] = p.z;
      this.vel[i * 3]     = p.vx;
      this.vel[i * 3 + 1] = p.vy;
      this.vel[i * 3 + 2] = p.vz;
      this.mass[i]        = p.mass ?? 0.001;
      this.lifetime[i]    = p.lifetime ?? 5.0;
      this.maxLife[i]     = this.lifetime[i];
      this.temp[i]        = p.temperature ?? 0.7;
      this.active[i]      = 1;
      this.count++;
    }
  }

  _findSlot() {
    // Try sequential first (fast path)
    for (let i = 0; i < MAX_SPH; i++) {
      if (!this.active[i]) return i;
    }
    return -1;
  }

  // ─── Step ─────────────────────────────────────────────────────────────────

  /**
   * @param {number} dt
   * @param {Function} externalAcc  (i, pos, out_acc) => void  e.g. gravity from bodies
   */
  step(dt, externalAcc) {
    const N = MAX_SPH;
    const h = this.h, h2 = this.h2, h6 = this.h6, h9 = this.h9;
    const rho0 = this.rho0, k = this.k, gamma = this.gamma;
    const visc = this.visc;
    const pos = this.pos, vel = this.vel, acc = this.acc;
    const mass = this.mass, density = this.density, pressure = this.pressure;
    const active = this.active;

    // Build active particle list
    const alive = [];
    for (let i = 0; i < N; i++) if (active[i]) alive.push(i);
    const aliveCount = alive.length;
    if (aliveCount === 0) return;

    // Density pass
    for (let ai = 0; ai < aliveCount; ai++) {
      const i = alive[ai];
      let rho = 0;
      const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
      for (let aj = 0; aj < aliveCount; aj++) {
        const j = alive[aj];
        const dx = px - pos[j * 3], dy = py - pos[j * 3 + 1], dz = pz - pos[j * 3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2) continue;
        const diff = h2 - r2;
        rho += mass[j] * POLY6_COEF / h9 * diff * diff * diff;
      }
      density[i] = rho;
      pressure[i] = k * (Math.pow(rho / rho0, gamma) - 1);
      if (pressure[i] < 0) pressure[i] = 0;
    }

    // Force pass
    for (let ai = 0; ai < aliveCount; ai++) {
      const i = alive[ai];
      let fx = 0, fy = 0, fz = 0;
      const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
      const rhoi = density[i] + 1e-8;
      const pi = pressure[i];

      for (let aj = 0; aj < aliveCount; aj++) {
        const j = alive[aj];
        if (i === j) continue;
        const dx = px - pos[j * 3], dy = py - pos[j * 3 + 1], dz = pz - pos[j * 3 + 2];
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= h2 || r2 < 1e-14) continue;
        const r = Math.sqrt(r2);
        const rhoj = density[j] + 1e-8;
        const pj = pressure[j];

        // Pressure force (spiky kernel gradient)
        const diff = h - r;
        const gradW = SPIKY_COEF / h6 * diff * diff / r;
        const pTerm = mass[j] * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj)) * gradW;
        fx -= pTerm * dx; fy -= pTerm * dy; fz -= pTerm * dz;

        // Artificial viscosity (Monaghan)
        const dvx = vel[i * 3] - vel[j * 3];
        const dvy = vel[i * 3 + 1] - vel[j * 3 + 1];
        const dvz = vel[i * 3 + 2] - vel[j * 3 + 2];
        const vDotR = dvx * dx + dvy * dy + dvz * dz;
        if (vDotR < 0) {
          const rhoAvg = (rhoi + rhoj) * 0.5;
          const mu = h * vDotR / (r2 + 0.01 * h2);
          const viscTerm = visc * mu / rhoAvg;
          // Poly6 laplacian approx via gradient magnitude
          const lapW = 45 / (Math.PI * h6) * (h - r);
          fx += mass[j] * viscTerm * lapW * dx / r;
          fy += mass[j] * viscTerm * lapW * dy / r;
          fz += mass[j] * viscTerm * lapW * dz / r;
        }
      }

      // External gravity
      if (externalAcc) {
        const tmpAcc = [0, 0, 0];
        externalAcc(i, pos, tmpAcc);
        fx += tmpAcc[0]; fy += tmpAcc[1]; fz += tmpAcc[2];
      }

      // Simple downward gravity
      fy -= this.gravity;

      acc[i * 3]     = fx;
      acc[i * 3 + 1] = fy;
      acc[i * 3 + 2] = fz;
    }

    // Integrate (symplectic Euler)
    for (let ai = 0; ai < aliveCount; ai++) {
      const i = alive[ai];
      vel[i * 3]     += acc[i * 3]     * dt;
      vel[i * 3 + 1] += acc[i * 3 + 1] * dt;
      vel[i * 3 + 2] += acc[i * 3 + 2] * dt;
      pos[i * 3]     += vel[i * 3]     * dt;
      pos[i * 3 + 1] += vel[i * 3 + 1] * dt;
      pos[i * 3 + 2] += vel[i * 3 + 2] * dt;

      // Lifetime decay
      this.lifetime[i] -= dt;
      if (this.lifetime[i] <= 0) {
        active[i] = 0;
        this.count = Math.max(0, this.count - 1);
      }
    }
  }

  /**
   * Get serializable snapshot for rendering thread.
   * Returns { positions:Float32Array, alphas:Float32Array, temps:Float32Array, count:number }
   */
  getSnapshot() {
    const alive = [];
    for (let i = 0; i < MAX_SPH; i++) if (this.active[i]) alive.push(i);
    const n = alive.length;
    const positions = new Float32Array(n * 3);
    const alphas    = new Float32Array(n);
    const temps     = new Float32Array(n);

    for (let ai = 0; ai < n; ai++) {
      const i = alive[ai];
      positions[ai * 3]     = this.pos[i * 3];
      positions[ai * 3 + 1] = this.pos[i * 3 + 1];
      positions[ai * 3 + 2] = this.pos[i * 3 + 2];
      alphas[ai]  = Math.min(1, this.lifetime[i] / Math.max(0.1, this.maxLife[i]));
      temps[ai]   = this.temp[i];
    }

    return { positions, alphas, temps, count: n };
  }

  clear() {
    this.count = 0;
    this.active.fill(0);
  }

  updateParams(h, k, gamma) {
    this.h = h; this.h2 = h * h; this.h6 = h * h * h * h * h * h;
    this.h9 = this.h6 * this.h2 * h;
    this.k = k; this.gamma = gamma;
  }
}
