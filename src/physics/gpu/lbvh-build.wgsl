// lbvh-build.wgsl
// WebGPU compute shaders for Linear BVH construction using Morton codes.
// Steps:
//   1. computeMorton: compute 30-bit Morton code for each body.
//   2. radixSort:     sort bodies by Morton code (4-bit radix, 8 passes).
//   3. buildTree:     construct LBVH internal nodes from sorted leaves.

// ─── Structs ───────────────────────────────────────────────────────────────────

struct AABB {
  minV : vec3<f32>,
  _p0  : f32,
  maxV : vec3<f32>,
  _p1  : f32,
};

struct LBVHNode {
  aabb      : AABB,
  leftChild : i32,   // index into nodes array; negative = leaf
  rightChild: i32,
  parent    : i32,
  bodyIdx   : i32,   // >= 0 for leaves
};

struct LBVHParams {
  N        : u32,
  sceneMin : vec3<f32>,
  _p0      : f32,
  sceneMax : vec3<f32>,
  _p1      : f32,
};

// ─── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       positions   : array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> mortonCodes : array<u32>;
@group(0) @binding(2) var<storage, read_write> sortedIdx   : array<u32>;
@group(0) @binding(3) var<storage, read_write> bvhNodes    : array<LBVHNode>;
@group(0) @binding(4) var<uniform>             lparams     : LBVHParams;

// ─── Morton Code ──────────────────────────────────────────────────────────────

fn expandBits(v: u32) -> u32 {
  var x = v & 0x3FFu;
  x = (x | (x << 16u)) & 0x030000FFu;
  x = (x | (x << 8u))  & 0x0300F00Fu;
  x = (x | (x << 4u))  & 0x030C30C3u;
  x = (x | (x << 2u))  & 0x09249249u;
  return x;
}

fn morton3D(ix: u32, iy: u32, iz: u32) -> u32 {
  return expandBits(ix) | (expandBits(iy) << 1u) | (expandBits(iz) << 2u);
}

@compute @workgroup_size(64)
fn computeMorton(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= lparams.N) { return; }

  let pos   = positions[i];
  let range = lparams.sceneMax - lparams.sceneMin;
  let scale = 1023.0 / max(range, vec3<f32>(1e-6));
  let fp    = clamp((pos - lparams.sceneMin) * scale, vec3<f32>(0.0), vec3<f32>(1023.0));
  let ix    = u32(fp.x);
  let iy    = u32(fp.y);
  let iz    = u32(fp.z);

  mortonCodes[i] = morton3D(ix, iy, iz);
  sortedIdx[i]   = i;
}

// ─── Radix Sort (single pass, 4-bit) ─────────────────────────────────────────
// (Full GPU radix sort would need multiple dispatches; this is a placeholder.
//  Real implementation would use 8 passes of 4-bit counting sort.)

var<workgroup> localHist : array<u32, 16>;

@compute @workgroup_size(64)
fn radixSortPass(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id)  lid: vec3<u32>) {
  // This pass is a stub — actual radix sort requires a scan (prefix sum)
  // which would be a separate shader. For now, CPU-side sort is used.
  // This shader is provided for the GPU path future implementation.
  if (lid.x == 0u) { localHist[0] = 0u; }
  workgroupBarrier();
}

// ─── LBVH Internal Node Construction ─────────────────────────────────────────
// Based on Karras 2012 "Maximizing Parallelism in the Construction of BVHs"

fn clz(x: u32) -> i32 {
  // Count leading zeros via bit scan
  var v = x;
  var c = 32i;
  if (v != 0u) {
    c = 0i;
    if ((v & 0xFFFF0000u) == 0u) { c += 16i; v <<= 16u; }
    if ((v & 0xFF000000u) == 0u) { c +=  8i; v <<= 8u;  }
    if ((v & 0xF0000000u) == 0u) { c +=  4i; v <<= 4u;  }
    if ((v & 0xC0000000u) == 0u) { c +=  2i; v <<= 2u;  }
    if ((v & 0x80000000u) == 0u) { c +=  1i; }
  }
  return c;
}

fn delta(i: i32, j: i32, N: i32) -> i32 {
  if (j < 0 || j >= N) { return -1i; }
  let a = mortonCodes[u32(i)];
  let b = mortonCodes[u32(j)];
  if (a == b) {
    // Tie-break by index
    return clz(a ^ b) + clz(u32(i) ^ u32(j));
  }
  return clz(a ^ b);
}

@compute @workgroup_size(64)
fn buildLBVHInternalNodes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  let N = i32(lparams.N);
  if (i >= N - 1) { return; }

  // Determine direction of range
  let d = sign(f32(delta(i, i + 1, N) - delta(i, i - 1, N)));
  let di = i32(d);

  // Compute upper bound for length of optimal range
  let dMin = delta(i, i - di, N);
  var lMax = 2i;
  while (delta(i, i + di * lMax, N) > dMin) { lMax *= 4i; }

  // Binary search for the end
  var l = 0i;
  var div = lMax / 2i;
  while (div >= 1i) {
    if (delta(i, i + di * (l + div), N) > dMin) { l += div; }
    div /= 2i;
  }
  let j = i + di * l;

  // Find split position
  let dNode = delta(i, j, N);
  var s = 0i;
  div = (abs(j - i) + 1i) / 2i;
  while (div >= 1i) {
    if (delta(i, i + di * (s + div), N) > dNode) { s += div; }
    div = max(div / 2i, 0i);
    if (div == 0i) { break; }
  }
  let gamma = i + di * s + min(di, 0i);

  // Left and right children
  let leftLeaf  = (min(i, j) == gamma);
  let rightLeaf = (max(i, j) == gamma + 1i);

  let nodeIdx = i; // internal nodes 0..N-2
  let leftChildIdx  = select(gamma,          N - 1 + gamma,         leftLeaf);
  let rightChildIdx = select(gamma + 1i, N - 1 + gamma + 1i, rightLeaf);

  bvhNodes[u32(nodeIdx)].leftChild  = leftChildIdx;
  bvhNodes[u32(nodeIdx)].rightChild = rightChildIdx;
  bvhNodes[u32(nodeIdx)].bodyIdx    = -1i;

  // Set parent pointers for children
  if (!leftLeaf) {
    bvhNodes[u32(leftChildIdx)].parent = nodeIdx;
  }
  if (!rightLeaf) {
    bvhNodes[u32(rightChildIdx)].parent = nodeIdx;
  }
}
