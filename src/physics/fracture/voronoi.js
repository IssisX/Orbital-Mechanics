/**
 * Event-driven Voronoi shatter system.
 * For large impacts, partitions a body into convex Voronoi cells.
 * Seeds are placed at high-strain regions (impact point vicinity).
 *
 * Algorithm:
 *   1. Generate N seed points biased toward impact zone.
 *   2. For each sub-particle: assign to nearest seed (Fortune's algorithm
 *      approximated via point-query for 3D; exact Voronoi would require
 *      a power diagram library, so we use nearest-seed classification).
 *   3. Each seed group → convex hull → new rigid fragment body.
 *   4. Mass proportional to number of particles in cell.
 */

import { RNG } from '../../util/rng.js';

export class VoronoiShatter {
  constructor() {
    this._rng = new RNG(0xDEADBEEF);
  }

  /**
   * Generate shatter fragments from body data.
   *
   * @param {Object} body  { cx, cy, cz, radius, mass, vel:[vx,vy,vz] }
   * @param {Object} impactPoint { x, y, z, normalX, normalY, normalZ }
   * @param {number} numFragments  Target fragment count [2..64]
   * @param {RNG}    rng
   * @returns {Object[]} Fragment descriptors:
   *   { cx, cy, cz, vx, vy, vz, mass, radius, stress }
   */
  shatter(body, impactPoint, numFragments, rng) {
    numFragments = Math.max(2, Math.min(numFragments | 0, 64));

    // Generate seed points biased toward impact zone
    const seeds = this._generateSeeds(body, impactPoint, numFragments, rng);

    // Scatter evaluation points inside body (stratified sphere)
    const evalCount = numFragments * 32;
    const evalPoints = this._sampleSphere(body, evalCount, rng);

    // Assign each eval point to nearest seed
    const assignment = new Int32Array(evalCount);
    for (let i = 0; i < evalCount; i++) {
      let minDist = Infinity, nearest = 0;
      const ex = evalPoints[i * 3], ey = evalPoints[i * 3 + 1], ez = evalPoints[i * 3 + 2];
      for (let s = 0; s < numFragments; s++) {
        const dx = seeds[s * 3] - ex, dy = seeds[s * 3 + 1] - ey, dz = seeds[s * 3 + 2] - ez;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < minDist) { minDist = d2; nearest = s; }
      }
      assignment[i] = nearest;
    }

    // Compute fragment properties
    const fragments = [];
    for (let s = 0; s < numFragments; s++) {
      let cx = 0, cy = 0, cz = 0, count = 0;
      for (let i = 0; i < evalCount; i++) {
        if (assignment[i] !== s) continue;
        cx += evalPoints[i * 3];
        cy += evalPoints[i * 3 + 1];
        cz += evalPoints[i * 3 + 2];
        count++;
      }
      if (count === 0) continue;
      cx /= count; cy /= count; cz /= count;

      const fracVol = count / evalCount;
      const fragMass = body.mass * fracVol;
      const fragRadius = body.radius * Math.cbrt(fracVol) * 1.1; // slight inflation

      // Fragment velocity: parent COM velocity + ejection from impact
      const seedX = seeds[s * 3], seedY = seeds[s * 3 + 1], seedZ = seeds[s * 3 + 2];
      const toSeedX = seedX - impactPoint.x;
      const toSeedY = seedY - impactPoint.y;
      const toSeedZ = seedZ - impactPoint.z;
      const toSeedLen = Math.sqrt(toSeedX ** 2 + toSeedY ** 2 + toSeedZ ** 2) || 1;

      // Ejection speed proportional to proximity to impact (nearest gets most energy)
      const distToImpact = Math.sqrt(
        (seedX - impactPoint.x) ** 2 + (seedY - impactPoint.y) ** 2 + (seedZ - impactPoint.z) ** 2
      );
      const ejectFactor = Math.max(0, 1 - distToImpact / (body.radius * 2));
      const ejectSpeed = ejectFactor * body.radius * 0.15; // heuristic

      // Stress = normalized proximity to impact point
      const stress = Math.exp(-distToImpact / (body.radius * 0.5));

      fragments.push({
        cx, cy, cz,
        vx: body.vel[0] + (toSeedX / toSeedLen) * ejectSpeed,
        vy: body.vel[1] + (toSeedY / toSeedLen) * ejectSpeed,
        vz: body.vel[2] + (toSeedZ / toSeedLen) * ejectSpeed,
        mass: fragMass,
        radius: Math.max(fragRadius, body.radius * 0.05),
        stress,
        type: 'fragment',
      });
    }

    return fragments;
  }

  _generateSeeds(body, impact, N, rng) {
    const seeds = new Float64Array(N * 3);
    // Half of seeds near impact point, half distributed in body
    const nearCount = Math.floor(N * 0.5);
    const spreadFactor = body.radius * 0.4;

    for (let s = 0; s < N; s++) {
      let sx, sy, sz;
      if (s < nearCount) {
        // Near impact zone, perturbed
        const [dx, dy, dz] = rng.inSphere(spreadFactor);
        sx = impact.x + dx;
        sy = impact.y + dy;
        sz = impact.z + dz;
      } else {
        // Distributed randomly inside body
        const [dx, dy, dz] = rng.inSphere(body.radius * 0.95);
        sx = body.cx + dx;
        sy = body.cy + dy;
        sz = body.cz + dz;
      }
      // Clamp to body sphere
      const dx = sx - body.cx, dy = sy - body.cy, dz = sz - body.cz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > body.radius) {
        const scale = body.radius * 0.9 / d;
        sx = body.cx + dx * scale;
        sy = body.cy + dy * scale;
        sz = body.cz + dz * scale;
      }
      seeds[s * 3] = sx; seeds[s * 3 + 1] = sy; seeds[s * 3 + 2] = sz;
    }
    return seeds;
  }

  _sampleSphere(body, N, rng) {
    const pts = new Float64Array(N * 3);
    for (let i = 0; i < N; i++) {
      const [dx, dy, dz] = rng.inSphere(body.radius);
      pts[i * 3] = body.cx + dx;
      pts[i * 3 + 1] = body.cy + dy;
      pts[i * 3 + 2] = body.cz + dz;
    }
    return pts;
  }

  /**
   * Generate ejecta SPH particle seeds from an impact.
   * Returns array of { x, y, z, vx, vy, vz, mass, lifetime } objects.
   */
  generateEjecta(body, impactPoint, impactSpeed, count, rng) {
    const ejecta = [];
    const nX = impactPoint.normalX || 0;
    const nY = impactPoint.normalY || 1;
    const nZ = impactPoint.normalZ || 0;

    // Build local frame from normal
    const tangX = Math.abs(nX) < 0.9 ? 1 : 0;
    const tangY = Math.abs(nX) < 0.9 ? 0 : 1;
    const tangZ = 0;

    const btX = nY * tangZ - nZ * tangY;
    const btY = nZ * tangX - nX * tangZ;
    const btZ = nX * tangY - nY * tangX;
    const btLen = Math.sqrt(btX * btX + btY * btY + btZ * btZ) || 1;

    for (let i = 0; i < count; i++) {
      // Cone of ejecta around impact normal
      const theta = rng.float(0, Math.PI * 2);
      const phi = rng.float(0, Math.PI * 0.45); // half-cone angle ~45°
      const sinPhi = Math.sin(phi), cosPhi = Math.cos(phi);

      const speed = impactSpeed * rng.float(0.05, 0.35);

      const evX = nX * cosPhi + (Math.cos(theta) * btX / btLen + Math.sin(theta) * (nY * btZ / btLen - nZ * btY / btLen)) * sinPhi;
      const evY = nY * cosPhi + (Math.cos(theta) * btY / btLen + Math.sin(theta) * (nZ * btX / btLen - nX * btZ / btLen)) * sinPhi;
      const evZ = nZ * cosPhi + (Math.cos(theta) * btZ / btLen + Math.sin(theta) * (nX * btY / btLen - nY * btX / btLen)) * sinPhi;

      ejecta.push({
        x: impactPoint.x + nX * body.radius * 0.02,
        y: impactPoint.y + nY * body.radius * 0.02,
        z: impactPoint.z + nZ * body.radius * 0.02,
        vx: body.vel[0] + evX * speed,
        vy: body.vel[1] + evY * speed,
        vz: body.vel[2] + evZ * speed,
        mass: body.mass * 1e-4,
        lifetime: rng.float(3.0, 8.0),
        temperature: rng.float(0.4, 1.0),
      });
    }
    return ejecta;
  }

  /**
   * Slice a body along a plane (for the slice tool).
   * Returns two halves.
   */
  sliceBody(body, planePoint, planeNormal, rng) {
    // Plane: dot(x - planePoint, planeNormal) = 0
    const nx = planeNormal[0], ny = planeNormal[1], nz = planeNormal[2];
    const d0 = nx * planePoint[0] + ny * planePoint[1] + nz * planePoint[2];

    // Split into positive and negative halves
    const halves = [[], []];
    const N = 64; // evaluation points
    const pts = this._sampleSphere(body, N, rng);

    for (let i = 0; i < N; i++) {
      const px = pts[i * 3], py = pts[i * 3 + 1], pz = pts[i * 3 + 2];
      const side = (nx * px + ny * py + nz * pz) > d0 ? 0 : 1;
      halves[side].push([px, py, pz]);
    }

    return halves.filter(h => h.length > 0).map((pts, idx) => {
      let cx = 0, cy = 0, cz = 0;
      for (const [x, y, z] of pts) { cx += x; cy += y; cz += z; }
      cx /= pts.length; cy /= pts.length; cz /= pts.length;
      const frac = pts.length / N;

      // Ejection from cut plane
      const ejSign = idx === 0 ? 1 : -1;
      const ejSpeed = 0.02 * body.radius;

      return {
        cx, cy, cz,
        vx: body.vel[0] + nx * ejSign * ejSpeed,
        vy: body.vel[1] + ny * ejSign * ejSpeed,
        vz: body.vel[2] + nz * ejSign * ejSpeed,
        mass: body.mass * frac,
        radius: body.radius * Math.cbrt(frac),
        stress: 0.5,
        type: 'fragment',
      };
    });
  }
}
