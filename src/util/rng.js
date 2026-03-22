/**
 * Seeded pseudo-random number generator using the Mulberry32 algorithm.
 * Deterministic: identical seed → identical sequence.
 * Period: ~4 billion values.
 */
export class RNG {
  constructor(seed = 42) {
    this.seed = (seed >>> 0) || 1;
    this._s = this.seed;
  }

  reset(seed) {
    this.seed = (seed >>> 0) || 1;
    this._s = this.seed;
  }

  /** Returns a float in [0, 1) */
  next() {
    let s = (this._s += 0x6D2B79F5);
    s = Math.imul(s ^ (s >>> 15), s | 1);
    s ^= s + Math.imul(s ^ (s >>> 7), s | 61);
    return ((s ^ (s >>> 14)) >>> 0) / 4294967296;
  }

  /** Float in [min, max) */
  float(min = 0, max = 1) {
    return min + this.next() * (max - min);
  }

  /** Integer in [min, max] (inclusive) */
  int(min, max) {
    return Math.floor(min + this.next() * (max - min + 1 - 1e-10));
  }

  /**
   * Normal distribution via Box-Muller transform.
   * @param {number} mean
   * @param {number} std  standard deviation
   */
  gaussian(mean = 0, std = 1) {
    const u = this.next() || 1e-10;
    const v = this.next();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Uniform random point on unit sphere, returns [x,y,z] */
  onSphere() {
    const theta = this.float(0, 2 * Math.PI);
    const phi = Math.acos(this.float(-1, 1));
    const sinPhi = Math.sin(phi);
    return [sinPhi * Math.cos(theta), sinPhi * Math.sin(theta), Math.cos(phi)];
  }

  /** Uniform random point inside sphere of given radius */
  inSphere(radius = 1) {
    const [x, y, z] = this.onSphere();
    const r = Math.cbrt(this.next()) * radius;
    return [x * r, y * r, z * r];
  }

  /** Uniform random point inside unit disk (xy) */
  inDisk() {
    const angle = this.float(0, 2 * Math.PI);
    const r = Math.sqrt(this.next());
    return [r * Math.cos(angle), r * Math.sin(angle)];
  }

  /** Shuffle an array in-place using Fisher-Yates */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(0, i);
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }
}

export const globalRNG = new RNG(42);
