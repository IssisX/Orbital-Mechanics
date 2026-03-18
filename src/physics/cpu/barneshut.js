/**
 * 3D Barnes–Hut octree gravity computation.
 * O(N log N) per step. Supports monopole gravity with softening.
 * Used as the CPU fallback when WebGPU is unavailable.
 */

const POOL_SIZE  = 16384;  // pre-allocated nodes; grows if needed
const MAX_DEPTH  = 48;     // guard against coplanar / degenerate inputs

class OctreeNode {
  constructor() {
    this.cx = 0; this.cy = 0; this.cz = 0; this.halfSize = 0;
    this.mass = 0;
    this.comX = 0; this.comY = 0; this.comZ = 0;
    this.bodyIdx = -1;   // ≥0 → leaf with one body; -1 → internal or empty
    this.ch = null;      // Array[8] of children when subdivided
  }
  reset() {
    this.cx = 0; this.cy = 0; this.cz = 0; this.halfSize = 0;
    this.mass = 0; this.comX = 0; this.comY = 0; this.comZ = 0;
    this.bodyIdx = -1; this.ch = null;
  }
}

export class BarnesHut {
  /**
   * @param {number} theta      Opening angle (default 0.6). Smaller = more accurate.
   * @param {number} epsilon    Gravitational softening length.
   * @param {number} G          Gravitational constant in simulation units.
   */
  constructor(theta = 0.6, epsilon = 0.05, G = 1.0) {
    this.theta    = theta;
    this.epsilon2 = epsilon * epsilon;
    this.G        = G;

    // Node object pool to avoid GC churn
    this._pool    = [];
    this._poolIdx = 0;
    for (let i = 0; i < POOL_SIZE; i++) this._pool.push(new OctreeNode());

    this.root = null;
  }

  _allocNode() {
    if (this._poolIdx >= this._pool.length) {
      this._pool.push(new OctreeNode());
    }
    const n = this._pool[this._poolIdx++];
    n.reset();
    return n;
  }

  /**
   * Build octree from flat body arrays.
   * @param {Float64Array} pos   stride-3 positions
   * @param {Float64Array} mass  body masses
   * @param {number} N
   */
  buildTree(pos, mass, N) {
    this._poolIdx = 0;
    this.root = null;

    if (N === 0) return;

    // Bounding box
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < N; i++) {
      const px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
      if (px < minX) minX = px; if (px > maxX) maxX = px;
      if (py < minY) minY = py; if (py > maxY) maxY = py;
      if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz;
    }

    // Cube half-size; add tiny epsilon to avoid degenerate zero-size cube
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const hs = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-8) * 0.5 * 1.001;

    this.root = this._allocNode();
    this.root.cx = cx; this.root.cy = cy; this.root.cz = cz;
    this.root.halfSize = hs;

    for (let i = 0; i < N; i++) {
      this._insert(this.root, pos, mass, i, 0);
    }

    this._computeMass(this.root, pos, mass);
  }

  _insert(node, pos, mass, idx, depth) {
    if (depth > MAX_DEPTH) return; // safety guard

    const px = pos[idx*3], py = pos[idx*3+1], pz = pos[idx*3+2];

    if (node.bodyIdx === -1 && node.ch === null) {
      // Empty leaf → store body here
      node.bodyIdx = idx;
      node.mass    = mass[idx];
      node.comX    = px; node.comY = py; node.comZ = pz;
      return;
    }

    if (node.ch === null) {
      // Non-empty leaf → subdivide; re-insert existing body downward
      const existIdx = node.bodyIdx;
      node.bodyIdx = -1;
      node.ch = [null, null, null, null, null, null, null, null];
      this._insertIntoChild(node, pos, mass, existIdx, depth + 1);
    }

    // Internal node: insert new body
    this._insertIntoChild(node, pos, mass, idx, depth + 1);
  }

  _insertIntoChild(node, pos, mass, idx, depth) {
    const px = pos[idx*3], py = pos[idx*3+1], pz = pos[idx*3+2];
    const hs2    = node.halfSize * 0.5;
    const xBit   = px >= node.cx ? 1 : 0;
    const yBit   = py >= node.cy ? 1 : 0;
    const zBit   = pz >= node.cz ? 1 : 0;
    const childI = xBit | (yBit << 1) | (zBit << 2);

    if (!node.ch[childI]) {
      const child    = this._allocNode();
      child.cx       = node.cx + (xBit ? hs2 : -hs2);
      child.cy       = node.cy + (yBit ? hs2 : -hs2);
      child.cz       = node.cz + (zBit ? hs2 : -hs2);
      child.halfSize = hs2;
      node.ch[childI] = child;
    }

    this._insert(node.ch[childI], pos, mass, idx, depth);
  }

  // Compute center of mass up from leaves. pos/mass needed for leaf initialization.
  _computeMass(node, pos, mass) {
    if (!node) return;

    if (node.bodyIdx !== -1) {
      // Leaf with a single body — already set in _insert
      return;
    }

    if (!node.ch) return;

    let totalMass = 0, cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < 8; i++) {
      const child = node.ch[i];
      if (!child) continue;
      this._computeMass(child, pos, mass);
      totalMass += child.mass;
      cx        += child.comX * child.mass;
      cy        += child.comY * child.mass;
      cz        += child.comZ * child.mass;
    }
    node.mass = totalMass;
    if (totalMass > 0) {
      node.comX = cx / totalMass;
      node.comY = cy / totalMass;
      node.comZ = cz / totalMass;
    }
  }

  /**
   * buildTree already calls _computeMass internally.
   * This is kept for API compatibility.
   */
  setLeafMasses(pos, mass, N) { /* no-op; handled in buildTree */ }

  /**
   * Compute gravitational accelerations for all N bodies.
   * @param {Float64Array} pos   stride-3 positions
   * @param {Float64Array} mass  body masses
   * @param {Float64Array} acc   Output accelerations stride-3 (reset here)
   * @param {number} N
   */
  computeAccelerations(pos, mass, acc, N) {
    acc.fill(0);
    if (!this.root || N === 0) return;

    const G      = this.G;
    const eps2   = this.epsilon2;
    const theta  = this.theta;
    const theta2 = theta * theta;

    // Per-body iterative traversal using an explicit stack
    const stack = new Array(256);

    for (let i = 0; i < N; i++) {
      const px = pos[i*3], py = pos[i*3+1], pz = pos[i*3+2];
      let ax = 0, ay = 0, az = 0;

      let sp = 0;
      stack[0] = this.root;
      sp = 1;

      while (sp > 0) {
        const node = stack[--sp];
        if (!node || node.mass === 0) continue;

        const dx = node.comX - px;
        const dy = node.comY - py;
        const dz = node.comZ - pz;
        const r2 = dx*dx + dy*dy + dz*dz + eps2;

        if (node.bodyIdx !== -1) {
          // Leaf — direct pairwise (skip self)
          if (node.bodyIdx === i) continue;
          const r3 = r2 * Math.sqrt(r2);
          const f  = G * node.mass / r3;
          ax += f * dx; ay += f * dy; az += f * dz;
        } else {
          // Barnes–Hut: compare (2*halfSize)² / r² < theta²
          const s = node.halfSize * 2;
          if ((s * s) <= theta2 * r2) {
            // Accept approximation
            const r3 = r2 * Math.sqrt(r2);
            const f  = G * node.mass / r3;
            ax += f * dx; ay += f * dy; az += f * dz;
          } else {
            // Descend into children
            if (node.ch) {
              for (let c = 0; c < 8; c++) {
                if (node.ch[c]) {
                  if (sp >= stack.length) stack.push(null);
                  stack[sp++] = node.ch[c];
                }
              }
            }
          }
        }
      }

      acc[i*3] = ax; acc[i*3+1] = ay; acc[i*3+2] = az;
    }
  }

  /**
   * Compute total gravitational potential energy (O(N²), used sparingly).
   */
  computePotentialEnergy(pos, mass, N) {
    const G = this.G, eps2 = this.epsilon2;
    let U = 0;
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = pos[j*3]-pos[i*3], dy = pos[j*3+1]-pos[i*3+1], dz = pos[j*3+2]-pos[i*3+2];
        const r  = Math.sqrt(dx*dx + dy*dy + dz*dz + eps2);
        U -= G * mass[i] * mass[j] / r;
      }
    }
    return U;
  }

  /**
   * Returns octree nodes for debug visualization.
   * @param {number} maxDepth
   */
  getDebugNodes(maxDepth = 3) {
    const nodes = [];
    if (!this.root) return nodes;
    this._collectNodes(this.root, 0, maxDepth, nodes);
    return nodes;
  }

  _collectNodes(node, depth, maxDepth, out) {
    if (!node || depth > maxDepth) return;
    out.push({ cx: node.cx, cy: node.cy, cz: node.cz,
               hs: node.halfSize, depth, mass: node.mass });
    if (node.ch) {
      for (let i = 0; i < 8; i++) {
        this._collectNodes(node.ch[i], depth + 1, maxDepth, out);
      }
    }
  }
}
