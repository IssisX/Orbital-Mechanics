// gravity-barneshut.wgsl
// WebGPU compute shader: Direct gravitational force summation (N-body).
// Used as fallback when Barnes-Hut tree GPU traversal is not implemented.
// For N < 2048, direct sum is fast enough on GPU.

struct Body {
  pos    : vec3<f32>,
  mass   : f32,
  vel    : vec3<f32>,
  radius : f32,
};

struct Params {
  bodyCount : u32,
  G         : f32,
  epsilon2  : f32,
  theta     : f32,
  dt        : f32,
  _pad0     : f32,
  _pad1     : f32,
  _pad2     : f32,
};

@group(0) @binding(0) var<storage, read>       bodies     : array<Body>;
@group(0) @binding(1) var<storage, read_write> accelerations : array<vec3<f32>>;
@group(0) @binding(2) var<uniform>             params     : Params;

// Shared memory tile for cache-efficient access
var<workgroup> tile : array<Body, 64>;

@compute @workgroup_size(64)
fn computeGravity(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(local_invocation_id)  lid : vec3<u32>,
  @builtin(workgroup_id)         wgid: vec3<u32>,
) {
  let i     = gid.x;
  let N     = params.bodyCount;
  let G     = params.G;
  let eps2  = params.epsilon2;

  var acc = vec3<f32>(0.0, 0.0, 0.0);

  if (i < N) {
    let pi = bodies[i].pos;

    // Tiled direct summation
    let numTiles = (N + 63u) / 64u;
    for (var t = 0u; t < numTiles; t++) {
      let j = t * 64u + lid.x;
      if (j < N) {
        tile[lid.x] = bodies[j];
      } else {
        tile[lid.x].mass = 0.0;
        tile[lid.x].pos  = vec3<f32>(0.0);
      }
      workgroupBarrier();

      for (var k = 0u; k < 64u; k++) {
        let jIdx = t * 64u + k;
        if (jIdx == i || jIdx >= N) { continue; }
        let d   = tile[k].pos - pi;
        let r2  = dot(d, d) + eps2;
        let r3  = r2 * sqrt(r2);
        acc    += (G * tile[k].mass / r3) * d;
      }
      workgroupBarrier();
    }
  }

  if (i < N) {
    accelerations[i] = acc;
  }
}

// ─── Leapfrog half-kick ────────────────────────────────────────────────────────

struct VelBuf {
  vel : array<vec3<f32>>,
};

@group(0) @binding(0) var<storage, read_write> velocities : array<vec3<f32>>;
@group(0) @binding(1) var<storage, read>       accs       : array<vec3<f32>>;
@group(0) @binding(2) var<uniform>             kp         : Params;

@compute @workgroup_size(64)
fn halfKick(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= kp.bodyCount) { return; }
  velocities[i] += accs[i] * (kp.dt * 0.5);
}

// ─── Drift ────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write> gpos : array<vec3<f32>>;
@group(0) @binding(1) var<storage, read>       gvel : array<vec3<f32>>;
@group(0) @binding(2) var<uniform>             dp   : Params;

@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dp.bodyCount) { return; }
  gpos[i] += gvel[i] * dp.dt;
}
