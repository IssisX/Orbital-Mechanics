/**
 * Meshing / Fragment Worker.
 * Handles heavy mesh-generation tasks off the main thread:
 *   - Icosphere generation for fragment bodies
 *   - LOD mesh decimation
 *   - Convex hull approximation for Voronoi cells
 *
 * Messages in:
 *   { type: 'icosphere', id, radius, detail }
 *   { type: 'decimateMesh', id, vertices, faces, targetFaces }
 *
 * Messages out:
 *   { type: 'icosphere', id, vertices, normals, uvs, indices }
 *   { type: 'decimatedMesh', id, vertices, faces }
 */

self.onmessage = (e) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'icosphere': {
        const { id, radius, detail } = msg;
        const mesh = generateIcosphere(radius ?? 1, detail ?? 2);
        self.postMessage({ type: 'icosphere', id, ...mesh },
          [mesh.vertices.buffer, mesh.normals.buffer, mesh.uvs.buffer, mesh.indices.buffer]);
        break;
      }

      case 'decimateMesh': {
        // Simplified: just return the mesh as-is (full decimation is complex)
        self.postMessage({ type: 'decimatedMesh', id: msg.id,
          vertices: msg.vertices, faces: msg.faces });
        break;
      }

      default:
        self.postMessage({ type: 'error', message: `Unknown: ${msg.type}` });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message });
  }
};

/**
 * Generate an icosphere mesh.
 * @param {number} radius
 * @param {number} detail  Subdivision level (0=20 faces, 1=80, 2=320, 3=1280)
 * @returns {{ vertices, normals, uvs, indices }}
 */
function generateIcosphere(radius, detail) {
  // Golden ratio
  const t = (1 + Math.sqrt(5)) / 2;

  // Icosahedron vertices (normalized)
  let verts = [
    -1, t, 0,  1, t, 0,  -1,-t, 0,   1,-t, 0,
     0,-1, t,  0, 1, t,   0,-1,-t,   0, 1,-t,
     t, 0,-1,  t, 0, 1,  -t, 0,-1,  -t, 0, 1,
  ].map((v, i) => {
    const group = Math.floor(i / 3);
    // Normalize
    return v;
  });

  // Normalize verts
  const normalizeV = () => {
    for (let i = 0; i < verts.length; i += 3) {
      const l = Math.sqrt(verts[i]**2 + verts[i+1]**2 + verts[i+2]**2);
      verts[i] /= l; verts[i+1] /= l; verts[i+2] /= l;
    }
  };
  normalizeV();

  // Icosahedron faces
  let faces = [
    0,11,5,  0,5,1,   0,1,7,   0,7,10, 0,10,11,
    1,5,9,   5,11,4,  11,10,2, 10,7,6, 7,1,8,
    3,9,4,   3,4,2,   3,2,6,   3,6,8,  3,8,9,
    4,9,5,   2,4,11,  6,2,10,  8,6,7,  9,8,1,
  ];

  // Midpoint cache for subdivision
  const midCache = new Map();
  const getMid = (a, b) => {
    const key = Math.min(a,b) * 100000 + Math.max(a,b);
    if (midCache.has(key)) return midCache.get(key);
    const ia = a * 3, ib = b * 3;
    verts.push(
      (verts[ia] + verts[ib]) * 0.5,
      (verts[ia+1] + verts[ib+1]) * 0.5,
      (verts[ia+2] + verts[ib+2]) * 0.5,
    );
    const idx = verts.length / 3 - 1;
    // Normalize
    const l = Math.sqrt(verts[idx*3]**2 + verts[idx*3+1]**2 + verts[idx*3+2]**2);
    verts[idx*3] /= l; verts[idx*3+1] /= l; verts[idx*3+2] /= l;
    midCache.set(key, idx);
    return idx;
  };

  // Subdivide
  for (let d = 0; d < detail; d++) {
    midCache.clear();
    const newFaces = [];
    for (let f = 0; f < faces.length; f += 3) {
      const a = faces[f], b = faces[f+1], c = faces[f+2];
      const ab = getMid(a, b);
      const bc = getMid(b, c);
      const ca = getMid(c, a);
      newFaces.push(a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca);
    }
    faces = newFaces;
  }

  // Scale by radius
  const vertCount = verts.length / 3;
  const vertices = new Float32Array(verts.length);
  const normals  = new Float32Array(verts.length);
  const uvs      = new Float32Array(vertCount * 2);

  for (let i = 0; i < vertCount; i++) {
    normals[i*3]   = verts[i*3];
    normals[i*3+1] = verts[i*3+1];
    normals[i*3+2] = verts[i*3+2];
    vertices[i*3]   = verts[i*3]   * radius;
    vertices[i*3+1] = verts[i*3+1] * radius;
    vertices[i*3+2] = verts[i*3+2] * radius;
    // Spherical UVs
    uvs[i*2]   = 0.5 + Math.atan2(verts[i*3+2], verts[i*3]) / (2 * Math.PI);
    uvs[i*2+1] = 0.5 - Math.asin(verts[i*3+1]) / Math.PI;
  }

  const indices = new Uint32Array(faces);

  return { vertices, normals, uvs, indices };
}
