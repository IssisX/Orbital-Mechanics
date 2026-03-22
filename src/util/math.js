/**
 * Lightweight 3D math utilities.
 * Plain-array API for minimal allocation. 'out' param pattern for hot paths.
 */

export const TAU = Math.PI * 2;
export const DEG2RAD = Math.PI / 180;
export const RAD2DEG = 180 / Math.PI;
export const EPSILON = 1e-10;

// ─── Vec3 ─────────────────────────────────────────────────────────────────────

export function v3set(out, x, y, z) { out[0] = x; out[1] = y; out[2] = z; return out; }
export function v3copy(a, out = [0, 0, 0]) { out[0] = a[0]; out[1] = a[1]; out[2] = a[2]; return out; }
export function v3zero(out = [0, 0, 0]) { out[0] = 0; out[1] = 0; out[2] = 0; return out; }

export function v3add(a, b, out = [0, 0, 0]) {
  out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2]; return out;
}
export function v3sub(a, b, out = [0, 0, 0]) {
  out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2]; return out;
}
export function v3scale(a, s, out = [0, 0, 0]) {
  out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s; return out;
}
export function v3addScaled(a, b, s, out = [0, 0, 0]) {
  out[0] = a[0] + b[0] * s; out[1] = a[1] + b[1] * s; out[2] = a[2] + b[2] * s; return out;
}
export function v3neg(a, out = [0, 0, 0]) {
  out[0] = -a[0]; out[1] = -a[1]; out[2] = -a[2]; return out;
}
export function v3dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function v3cross(a, b, out = [0, 0, 0]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
  return out;
}
export function v3len(a) { return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
export function v3len2(a) { return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }
export function v3dist(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
export function v3dist2(a, b) {
  const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
  return dx * dx + dy * dy + dz * dz;
}
export function v3normalize(a, out = [0, 0, 0]) {
  const l = v3len(a);
  if (l < 1e-300) { out[0] = out[1] = out[2] = 0; return out; }
  return v3scale(a, 1 / l, out);
}
export function v3lerp(a, b, t, out = [0, 0, 0]) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}
export function v3mul(a, b, out = [0, 0, 0]) {
  out[0] = a[0] * b[0]; out[1] = a[1] * b[1]; out[2] = a[2] * b[2]; return out;
}

// ─── Scalar ────────────────────────────────────────────────────────────────────

export function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
export function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
export function lerp(a, b, t) { return a + (b - a) * t; }
export function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}
export function sign(x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

// ─── Morton Codes (3D, 10 bits per axis → 30 bit code) ───────────────────────

function expandBits10(v) {
  v = (v | (v << 16)) & 0x030000FF;
  v = (v | (v << 8)) & 0x0300F00F;
  v = (v | (v << 4)) & 0x030C30C3;
  v = (v | (v << 2)) & 0x09249249;
  return v;
}

/**
 * Compute 30-bit Morton code from integer coordinates [0,1023].
 */
export function morton3D(ix, iy, iz) {
  return (expandBits10(ix & 0x3FF) |
         (expandBits10(iy & 0x3FF) << 1) |
         (expandBits10(iz & 0x3FF) << 2)) >>> 0;
}

/**
 * Map a 3D world position to a Morton code given scene bounds.
 * @param {number} px,py,pz  world position
 * @param {number} minV       scene min (uniform)
 * @param {number} maxV       scene max (uniform)
 */
export function mortonFromPos(px, py, pz, minV, maxV) {
  const scale = 1023 / Math.max(maxV - minV, 1e-10);
  const ix = clamp(((px - minV) * scale) | 0, 0, 1023);
  const iy = clamp(((py - minV) * scale) | 0, 0, 1023);
  const iz = clamp(((pz - minV) * scale) | 0, 0, 1023);
  return morton3D(ix, iy, iz);
}

// ─── AABB ─────────────────────────────────────────────────────────────────────

/** Expand AABB by point [px,py,pz]. out = [minX,minY,minZ,maxX,maxY,maxZ] */
export function aabbExpand(aabb, px, py, pz) {
  if (px < aabb[0]) aabb[0] = px;
  if (py < aabb[1]) aabb[1] = py;
  if (pz < aabb[2]) aabb[2] = pz;
  if (px > aabb[3]) aabb[3] = px;
  if (py > aabb[4]) aabb[4] = py;
  if (pz > aabb[5]) aabb[5] = pz;
}

export function aabbUnion(a, b, out = new Float64Array(6)) {
  out[0] = Math.min(a[0], b[0]); out[1] = Math.min(a[1], b[1]); out[2] = Math.min(a[2], b[2]);
  out[3] = Math.max(a[3], b[3]); out[4] = Math.max(a[4], b[4]); out[5] = Math.max(a[5], b[5]);
  return out;
}

export function aabbSurface(aabb) {
  const dx = aabb[3] - aabb[0], dy = aabb[4] - aabb[1], dz = aabb[5] - aabb[2];
  return 2 * (dx * dy + dy * dz + dz * dx);
}

export function aabbOverlap(a, b) {
  return a[3] >= b[0] && a[0] <= b[3] &&
         a[4] >= b[1] && a[1] <= b[4] &&
         a[5] >= b[2] && a[2] <= b[5];
}

// ─── Quaternion ───────────────────────────────────────────────────────────────
// Quaternion stored as [x, y, z, w]

export function qIdentity(out = [0, 0, 0, 1]) {
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1; return out;
}

export function qMul(a, b, out = [0, 0, 0, 1]) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function qNormalize(q, out = [0, 0, 0, 1]) {
  const l = Math.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if (l < 1e-300) { out[0]=0;out[1]=0;out[2]=0;out[3]=1; return out; }
  const il = 1 / l;
  out[0] = q[0]*il; out[1] = q[1]*il; out[2] = q[2]*il; out[3] = q[3]*il;
  return out;
}

export function qFromAxisAngle(axis, angle, out = [0, 0, 0, 1]) {
  const half = angle * 0.5;
  const s = Math.sin(half);
  const norm = v3len(axis);
  if (norm < 1e-10) { return qIdentity(out); }
  const inv = s / norm;
  out[0] = axis[0] * inv; out[1] = axis[1] * inv; out[2] = axis[2] * inv;
  out[3] = Math.cos(half);
  return out;
}

export function qRotateVec3(q, v, out = [0, 0, 0]) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (qy * vz - qz * vy);
  const ty = 2 * (qz * vx - qx * vz);
  const tz = 2 * (qx * vy - qy * vx);
  out[0] = vx + qw * tx + qy * tz - qz * ty;
  out[1] = vy + qw * ty + qz * tx - qx * tz;
  out[2] = vz + qw * tz + qx * ty - qy * tx;
  return out;
}

// ─── Misc ─────────────────────────────────────────────────────────────────────

/** Sphere volume */
export function sphereVolume(r) { return (4 / 3) * Math.PI * r * r * r; }

/** Uniform density mass from radius */
export function massFromRadius(r, density = 1) { return density * sphereVolume(r); }

/** Sphere inertia tensor diagonal for axis */
export function sphereInertia(mass, r) { return 0.4 * mass * r * r; }

/** Gravitational potential energy between two bodies */
export function gravitationalPE(G, m1, m2, dist) { return -G * m1 * m2 / Math.max(dist, 1e-10); }

/** Kinetic energy */
export function kineticEnergy(mass, vx, vy, vz) {
  return 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}
