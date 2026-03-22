// neighbors.wgsl
// WebGPU compute shader: Query k-nearest neighbors from LBVH.
// Used for SPH density estimation and peridynamics bond queries on GPU path.

struct NeighborParams {
  N           : u32,
  queryCount  : u32,
  searchRadius: f32,
  maxNeighbors: u32,
};

struct AABB {
  minV : vec3<f32>,
  _p0  : f32,
  maxV : vec3<f32>,
  _p1  : f32,
};

struct LBVHNode {
  aabb       : AABB,
  leftChild  : i32,
  rightChild : i32,
  parent     : i32,
  bodyIdx    : i32,
};

@group(0) @binding(0) var<storage, read>       queryPos     : array<vec3<f32>>;
@group(0) @binding(1) var<storage, read>       bvhNodes     : array<LBVHNode>;
@group(0) @binding(2) var<storage, read>       sortedPos    : array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> neighborList : array<i32>;
@group(0) @binding(4) var<storage, read_write> neighborCount: array<u32>;
@group(0) @binding(5) var<uniform>             nparams      : NeighborParams;

fn aabbOverlapSphere(aabb: AABB, center: vec3<f32>, r: f32) -> bool {
  let clamped = clamp(center, aabb.minV, aabb.maxV);
  let d = center - clamped;
  return dot(d, d) <= r * r;
}

// Stack-based BVH traversal
var<private> stack: array<i32, 64>;

@compute @workgroup_size(64)
fn queryNeighbors(@builtin(global_invocation_id) gid: vec3<u32>) {
  let qi = gid.x;
  if (qi >= nparams.queryCount) { return; }

  let qpos  = queryPos[qi];
  let r     = nparams.searchRadius;
  let maxNb = nparams.maxNeighbors;
  let base  = qi * maxNb;

  var count = 0u;
  var sp    = 0i;
  stack[0]  = 0i; // root
  sp = 1i;

  while (sp > 0i) {
    sp -= 1i;
    let nodeIdx = stack[sp];
    if (nodeIdx < 0i) { continue; }
    let node = bvhNodes[u32(nodeIdx)];

    if (!aabbOverlapSphere(node.aabb, qpos, r)) { continue; }

    if (node.bodyIdx >= 0i) {
      // Leaf
      let bp  = sortedPos[u32(node.bodyIdx)];
      let d   = qpos - bp;
      if (dot(d, d) <= r * r && count < maxNb) {
        neighborList[base + count] = node.bodyIdx;
        count++;
      }
    } else {
      // Internal: push children
      if (sp < 62i) {
        if (node.leftChild  >= 0i) { stack[sp] = node.leftChild;  sp++; }
        if (node.rightChild >= 0i) { stack[sp] = node.rightChild; sp++; }
      }
    }
  }

  neighborCount[qi] = count;
  // Pad remaining slots with -1
  for (var k = count; k < maxNb; k++) {
    neighborList[base + k] = -1i;
  }
}

// ─── SPH Density Compute ──────────────────────────────────────────────────────

struct SPHParams {
  N       : u32,
  h       : f32,
  h2      : f32,
  h9      : f32,
  rho0    : f32,
  k       : f32,
  gamma   : f32,
  _pad    : f32,
};

@group(0) @binding(0) var<storage, read>       sphPos     : array<vec3<f32>>;
@group(0) @binding(1) var<storage, read>       sphMass    : array<f32>;
@group(0) @binding(2) var<storage, read>       sphNbList  : array<i32>;
@group(0) @binding(3) var<storage, read>       sphNbCount : array<u32>;
@group(0) @binding(4) var<storage, read_write> sphDensity : array<f32>;
@group(0) @binding(5) var<storage, read_write> sphPressure: array<f32>;
@group(0) @binding(6) var<uniform>             sparams    : SPHParams;

const POLY6_COEF : f32 = 315.0 / (64.0 * 3.14159265359);

@compute @workgroup_size(64)
fn sphDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= sparams.N) { return; }

  let pi   = sphPos[i];
  let h2   = sparams.h2;
  let h9   = sparams.h9;
  let base = i * 32u; // max 32 neighbors per particle
  let nb   = sphNbCount[i];

  var rho = 0.0;
  for (var k = 0u; k < nb; k++) {
    let j = sphNbList[base + k];
    if (j < 0i) { break; }
    let d   = pi - sphPos[u32(j)];
    let r2  = dot(d, d);
    if (r2 >= h2) { continue; }
    let diff = h2 - r2;
    rho += sphMass[u32(j)] * POLY6_COEF / h9 * diff * diff * diff;
  }

  sphDensity[i]  = rho;
  sphPressure[i] = sparams.k * (pow(rho / sparams.rho0, sparams.gamma) - 1.0);
  if (sphPressure[i] < 0.0) { sphPressure[i] = 0.0; }
}
