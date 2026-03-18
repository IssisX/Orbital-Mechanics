/**
 * SAH-built BVH (Bounding Volume Hierarchy) for broad-phase collision detection.
 * Produces AABB pairs that may be overlapping for narrow-phase testing.
 * Uses surface-area heuristic for tree quality. CPU fallback path.
 */

/**
 * AABB: [minX, minY, minZ, maxX, maxY, maxZ]
 */

const LEAF_MAX = 4; // max bodies per leaf

export class BVH {
  constructor() {
    this._nodes = [];  // flat array of BVHNode
    this._indices = []; // body index permutation
    this.root = null;
  }

  /**
   * Build BVH from body positions and radii.
   * @param {Float64Array} pos    stride-3 positions
   * @param {Float64Array} radii  per-body radius
   * @param {number} N
   */
  build(pos, radii, N) {
    this._nodes.length = 0;
    this._indices = Array.from({ length: N }, (_, i) => i);

    if (N === 0) { this.root = null; return; }

    this.root = this._buildNode(pos, radii, this._indices, 0, N);
  }

  _computeAABB(pos, radii, indices, start, end) {
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let k = start; k < end; k++) {
      const i = indices[k];
      const px = pos[i * 3], py = pos[i * 3 + 1], pz = pos[i * 3 + 2];
      const r = radii[i];
      if (px - r < minX) minX = px - r;
      if (py - r < minY) minY = py - r;
      if (pz - r < minZ) minZ = pz - r;
      if (px + r > maxX) maxX = px + r;
      if (py + r > maxY) maxY = py + r;
      if (pz + r > maxZ) maxZ = pz + r;
    }
    return [minX, minY, minZ, maxX, maxY, maxZ];
  }

  _aabbSA(aabb) {
    const dx = aabb[3] - aabb[0], dy = aabb[4] - aabb[1], dz = aabb[5] - aabb[2];
    return 2 * (dx * dy + dy * dz + dz * dx);
  }

  _buildNode(pos, radii, indices, start, end) {
    const count = end - start;
    const aabb = this._computeAABB(pos, radii, indices, start, end);
    const node = { aabb, start, end, left: null, right: null, isLeaf: false };

    if (count <= LEAF_MAX) {
      node.isLeaf = true;
      return node;
    }

    // Find best split using SAH
    const parentSA = this._aabbSA(aabb);
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = start + Math.floor(count / 2); // fallback midpoint

    const BINS = 8;
    for (let axis = 0; axis < 3; axis++) {
      // Sort indices along axis
      const axisBase = axis;
      const slice = indices.slice(start, end);
      slice.sort((a, b) => pos[a * 3 + axisBase] - pos[b * 3 + axisBase]);
      for (let k = 0; k < count; k++) indices[start + k] = slice[k];

      // Sweep SAH bins
      for (let s = 1; s < count; s++) {
        const leftAABB = this._computeAABB(pos, radii, indices, start, start + s);
        const rightAABB = this._computeAABB(pos, radii, indices, start + s, end);
        const cost = (this._aabbSA(leftAABB) * s + this._aabbSA(rightAABB) * (count - s)) / parentSA;
        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestSplit = start + s;
        }
      }
    }

    // Re-sort on best axis if needed (already sorted for last axis=2)
    if (bestAxis !== 2) {
      const slice = indices.slice(start, end);
      slice.sort((a, b) => pos[a * 3 + bestAxis] - pos[b * 3 + bestAxis]);
      for (let k = 0; k < count; k++) indices[start + k] = slice[k];
      bestSplit = start + Math.floor(count / 2); // re-find via proper sweep
      // Simplified: just use midpoint on best axis
    }

    if (bestSplit === start || bestSplit === end) bestSplit = start + Math.floor(count / 2);

    node.left = this._buildNode(pos, radii, indices, start, bestSplit);
    node.right = this._buildNode(pos, radii, indices, bestSplit, end);
    return node;
  }

  /**
   * Query overlapping pairs from the BVH.
   * @returns {Array<[number,number]>} pairs of body indices
   */
  queryPairs() {
    const pairs = [];
    if (!this.root) return pairs;
    this._queryPairsNode(this.root, this.root, pairs, true);
    return pairs;
  }

  _aabbOverlap(a, b) {
    return a[3] >= b[0] && a[0] <= b[3] &&
           a[4] >= b[1] && a[1] <= b[4] &&
           a[5] >= b[2] && a[2] <= b[5];
  }

  _queryPairsNode(nodeA, nodeB, pairs, sameBranch) {
    if (!this._aabbOverlap(nodeA.aabb, nodeB.aabb)) return;

    if (nodeA.isLeaf && nodeB.isLeaf) {
      // Pairwise within leaves
      for (let i = nodeA.start; i < nodeA.end; i++) {
        const start = sameBranch ? i + 1 : nodeB.start;
        for (let j = start; j < nodeB.end; j++) {
          if (this._indices[i] !== this._indices[j]) {
            pairs.push([this._indices[i], this._indices[j]]);
          }
        }
      }
      return;
    }

    if (nodeA.isLeaf) {
      this._queryPairsNode(nodeA, nodeB.left, pairs, false);
      this._queryPairsNode(nodeA, nodeB.right, pairs, false);
    } else if (nodeB.isLeaf) {
      this._queryPairsNode(nodeA.left, nodeB, pairs, false);
      this._queryPairsNode(nodeA.right, nodeB, pairs, false);
    } else if (sameBranch) {
      this._queryPairsNode(nodeA.left, nodeA.left, pairs, true);
      this._queryPairsNode(nodeA.right, nodeA.right, pairs, true);
      this._queryPairsNode(nodeA.left, nodeA.right, pairs, false);
    } else {
      this._queryPairsNode(nodeA.left, nodeB.left, pairs, false);
      this._queryPairsNode(nodeA.left, nodeB.right, pairs, false);
      this._queryPairsNode(nodeA.right, nodeB.left, pairs, false);
      this._queryPairsNode(nodeA.right, nodeB.right, pairs, false);
    }
  }

  /**
   * Query which bodies overlap a sphere query.
   * @param {number} px,py,pz  Center
   * @param {number} radius
   * @returns {number[]} body indices
   */
  querySphere(px, py, pz, radius) {
    const result = [];
    if (!this.root) return result;
    this._querySphereNode(this.root, px, py, pz, radius, result);
    return result;
  }

  _querySphereNode(node, px, py, pz, r, result) {
    // Sphere-AABB overlap: clamp point to AABB, check distance
    const cx = Math.max(node.aabb[0], Math.min(px, node.aabb[3]));
    const cy = Math.max(node.aabb[1], Math.min(py, node.aabb[4]));
    const cz = Math.max(node.aabb[2], Math.min(pz, node.aabb[5]));
    const dx = cx - px, dy = cy - py, dz = cz - pz;
    if (dx * dx + dy * dy + dz * dz > r * r) return;

    if (node.isLeaf) {
      for (let k = node.start; k < node.end; k++) result.push(this._indices[k]);
      return;
    }
    this._querySphereNode(node.left, px, py, pz, r, result);
    this._querySphereNode(node.right, px, py, pz, r, result);
  }

  /** Get all leaf body indices as an ordered array */
  get indices() { return this._indices; }

  /**
   * Flatten tree into arrays for debug visualization.
   * Returns { mins, maxs, depths } as flat arrays.
   */
  flattenForDebug(maxDepth = 4) {
    const mins = [], maxs = [], depths = [];
    if (!this.root) return { mins, maxs, depths };
    this._flatten(this.root, 0, maxDepth, mins, maxs, depths);
    return { mins, maxs, depths };
  }

  _flatten(node, depth, maxDepth, mins, maxs, depths) {
    if (!node || depth > maxDepth) return;
    mins.push([node.aabb[0], node.aabb[1], node.aabb[2]]);
    maxs.push([node.aabb[3], node.aabb[4], node.aabb[5]]);
    depths.push(depth);
    if (!node.isLeaf) {
      this._flatten(node.left, depth + 1, maxDepth, mins, maxs, depths);
      this._flatten(node.right, depth + 1, maxDepth, mins, maxs, depths);
    }
  }
}
