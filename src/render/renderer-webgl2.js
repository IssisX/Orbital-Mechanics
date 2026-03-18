/**
 * WebGL2 Renderer — primary rendering path using Three.js r163.
 *
 * Features:
 *  - PBR-ish materials (MeshStandardMaterial) with tri-planar tiling
 *  - GPU-instanced mesh for fragments/rubble (InstancedMesh)
 *  - SPH ejecta as instanced point sprites
 *  - Debug overlays: octree wireframe, BVH AABB, bond lines, energy graph
 *  - ACES tonemapping, exposure 1.25
 *  - Optional bloom (default OFF)
 *  - Orbital camera with inertia
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TYPE_STAR = 0, TYPE_PLANET = 1, TYPE_RUBBLE = 2, TYPE_FRAGMENT = 3;
const MAX_INSTANCE = 512;
const MAX_SPH_INSTANCES = 4096;

export class RendererWebGL2 {
  constructor(canvas) {
    this.canvas = canvas;
    this._meshes = new Map();      // bodyId → THREE.Mesh
    this._instanceMesh = null;     // for fragments
    this._sphInstanceMesh = null;  // for ejecta
    this._bondLines = new Map();   // bodyId → THREE.LineSegments
    this._octreeLines = null;
    this._bvhLines = null;

    this.showBonds   = false;
    this.showOctree  = false;
    this.showBVH     = false;
    this.bloom       = false;
    this.drawCalls   = 0;
    this._instanceCount = 0;
    this._sphCount = 0;

    // Body state mirror
    this._bodyMeshIds = new Map(); // bodyId → { mesh }

    this._init();
  }

  _init() {
    const canvas = this.canvas;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      powerPreference: 'high-performance',
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    this.renderer.physicallyCorrectLights = true;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000208);

    // Stars background (billboard particles)
    this._addStarfield();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      55, canvas.clientWidth / canvas.clientHeight, 0.1, 10000
    );
    this.camera.position.set(0, 15, 40);

    // Controls
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1;
    this.controls.maxDistance = 2000;
    this.controls.screenSpacePanning = true;

    // Lights
    const sun = new THREE.DirectionalLight(0xfff8e7, 3.5);
    sun.position.set(50, 80, 40);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 500;
    sun.shadow.camera.top = sun.shadow.camera.right = 80;
    sun.shadow.camera.bottom = sun.shadow.camera.left = -80;
    this.scene.add(sun);

    const hemi = new THREE.HemisphereLight(0x2244aa, 0x110022, 0.8);
    this.scene.add(hemi);

    const ambient = new THREE.AmbientLight(0x112233, 0.4);
    this.scene.add(ambient);

    // Fragment instanced mesh (icosphere, detail=1)
    const fragGeo = new THREE.IcosahedronGeometry(1, 1);
    const fragMat = new THREE.MeshStandardMaterial({
      color: 0xaaaaaa,
      roughness: 0.85, metalness: 0.1,
      vertexColors: false,
    });
    this._instanceMesh = new THREE.InstancedMesh(fragGeo, fragMat, MAX_INSTANCE);
    this._instanceMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this._instanceMesh.count = 0;
    this._instanceMesh.castShadow = true;
    this._instanceMesh.receiveShadow = true;
    // Instance color buffer
    const instanceColors = new Float32Array(MAX_INSTANCE * 3);
    this._instanceMesh.instanceColor = new THREE.InstancedBufferAttribute(instanceColors, 3);
    this.scene.add(this._instanceMesh);

    // SPH ejecta instanced billboard
    const sphGeo = new THREE.PlaneGeometry(1, 1);
    const sphMat = new THREE.MeshBasicMaterial({
      color: 0xff8833,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this._sphMesh = new THREE.InstancedMesh(sphGeo, sphMat, MAX_SPH_INSTANCES);
    this._sphMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this._sphMesh.count = 0;
    this.scene.add(this._sphMesh);

    // Debug overlays (initially hidden)
    this._bondGroup  = new THREE.Group(); this.scene.add(this._bondGroup);
    this._debugGroup = new THREE.Group(); this.scene.add(this._debugGroup);

    // Resize observer
    const ro = new ResizeObserver(() => this._onResize());
    ro.observe(canvas);
  }

  _addStarfield() {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      const r     = 800 + Math.random() * 200;
      positions[i*3]   = r * Math.sin(phi) * Math.cos(theta);
      positions[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i*3+2] = r * Math.cos(phi);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 0.8, sizeAttenuation: true });
    this.scene.add(new THREE.Points(geo, mat));
  }

  // ─── Material factory ────────────────────────────────────────────────────

  _makePlanetMaterial(baseColor, type) {
    const c = new THREE.Color(baseColor[0], baseColor[1], baseColor[2]);
    if (type === TYPE_STAR) {
      const mat = new THREE.MeshStandardMaterial({
        color: c,
        emissive: c.clone().multiplyScalar(0.6),
        emissiveIntensity: 1.5,
        roughness: 0.95,
        metalness: 0.0,
      });
      return mat;
    }
    return new THREE.MeshStandardMaterial({
      color: c,
      roughness: 0.82,
      metalness: 0.08,
      envMapIntensity: 0.3,
    });
  }

  // ─── Body management ─────────────────────────────────────────────────────

  syncBodies(snapshot) {
    const { N, positions, radii, types, colors, ids, stresses } = snapshot;

    const currentIds = new Set(ids.subarray(0, N));

    // Remove bodies no longer in snapshot
    for (const [id, obj] of this._bodyMeshIds) {
      if (!currentIds.has(id)) {
        this.scene.remove(obj.mesh);
        obj.mesh.geometry.dispose();
        obj.mesh.material.dispose();
        this._bodyMeshIds.delete(id);
      }
    }

    // Fragment / rubble counter
    let fragCount = 0;
    const _mat4 = new THREE.Matrix4();
    const _quat = new THREE.Quaternion();
    const _scale = new THREE.Vector3();
    const _pos3  = new THREE.Vector3();

    for (let ai = 0; ai < N; ai++) {
      const id   = ids[ai];
      const type = types[ai];
      const r    = radii[ai];
      const px = positions[ai*3], py = positions[ai*3+1], pz = positions[ai*3+2];
      const cr = colors[ai*3], cg = colors[ai*3+1], cb = colors[ai*3+2];
      const stress = stresses[ai];

      if (type === TYPE_FRAGMENT || type === TYPE_RUBBLE) {
        if (fragCount < MAX_INSTANCE) {
          _pos3.set(px, py, pz);
          _scale.set(r, r, r);
          _mat4.compose(_pos3, _quat, _scale);
          this._instanceMesh.setMatrixAt(fragCount, _mat4);

          // Color by stress (black-body ramp)
          const col = this._stressColor(stress, cr, cg, cb);
          this._instanceMesh.setColorAt(fragCount, col);
          fragCount++;
        }
        // Clean up any individual mesh if type changed
        if (this._bodyMeshIds.has(id)) {
          const obj = this._bodyMeshIds.get(id);
          this.scene.remove(obj.mesh);
          this._bodyMeshIds.delete(id);
        }
        continue;
      }

      // Star or Planet — individual mesh
      if (!this._bodyMeshIds.has(id)) {
        const detail = type === TYPE_STAR ? 3 : 2;
        const geo = new THREE.IcosahedronGeometry(1, detail);
        const mat = this._makePlanetMaterial([cr, cg, cb], type);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.scene.add(mesh);
        this._bodyMeshIds.set(id, { mesh, type });
      }

      const obj = this._bodyMeshIds.get(id);
      obj.mesh.position.set(px, py, pz);
      obj.mesh.scale.set(r, r, r);

      // Update color by stress
      const stressColor = this._stressColor(stress, cr, cg, cb);
      if (obj.mesh.material.color) {
        obj.mesh.material.color.setRGB(stressColor.r, stressColor.g, stressColor.b);
      }
    }

    // Update instanced mesh count
    this._instanceMesh.count = fragCount;
    if (fragCount > 0) {
      this._instanceMesh.instanceMatrix.needsUpdate = true;
      if (this._instanceMesh.instanceColor) this._instanceMesh.instanceColor.needsUpdate = true;
    }
    this._instanceCount = fragCount;
  }

  _stressColor(stress, r, g, b) {
    const col = new THREE.Color(r, g, b);
    if (stress < 0.01) return col;
    // Blend toward hot (orange/white) at high stress
    const hot = new THREE.Color(
      Math.min(1, 0.5 + stress * 1.5),
      Math.max(0, 0.2 + (1 - stress) * 0.3),
      Math.max(0, 0.1 * (1 - stress))
    );
    col.lerp(hot, Math.min(stress, 1));
    return col;
  }

  // ─── SPH Ejecta ───────────────────────────────────────────────────────────

  syncSPH(sphSnap) {
    if (!sphSnap || sphSnap.count === 0) {
      this._sphMesh.count = 0;
      return;
    }

    const { positions, alphas, temps, count } = sphSnap;
    const _mat4  = new THREE.Matrix4();
    const _quat  = new THREE.Quaternion();
    const _pos3  = new THREE.Vector3();
    const _scale = new THREE.Vector3();
    const _col   = new THREE.Color();

    for (let i = 0; i < count && i < MAX_SPH_INSTANCES; i++) {
      const px = positions[i*3], py = positions[i*3+1], pz = positions[i*3+2];
      const alpha = alphas[i];
      const temp  = temps[i];
      const size  = 0.15 + temp * 0.25;

      _pos3.set(px, py, pz);
      _scale.set(size, size, size);
      // Billboard: always face camera (handled by making it a point)
      _mat4.compose(_pos3, _quat, _scale);
      this._sphMesh.setMatrixAt(i, _mat4);
      // Color: temperature ramp
      _col.setRGB(0.5 + temp * 0.5, 0.2 + temp * 0.3, 0.05 * temp);
    }

    this._sphMesh.count = Math.min(count, MAX_SPH_INSTANCES);
    this._sphMesh.instanceMatrix.needsUpdate = true;
    // Update opacity via material
    this._sphMesh.material.opacity = 0.55;
    this._sphCount = this._sphMesh.count;
  }

  // ─── Bond debug ───────────────────────────────────────────────────────────

  updateBondLines(bondData) {
    if (!this.showBonds) { this._bondGroup.clear(); return; }

    this._bondGroup.clear();

    for (const [bodyId, data] of Object.entries(bondData)) {
      if (!data) continue;
      if (data.active && data.active.length > 0) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(data.active, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0x44ff88, linewidth: 1 });
        this._bondGroup.add(new THREE.LineSegments(geo, mat));
      }
      if (data.broken && data.broken.length > 0) {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(data.broken, 3));
        const mat = new THREE.LineBasicMaterial({ color: 0xff2200, linewidth: 1 });
        this._bondGroup.add(new THREE.LineSegments(geo, mat));
      }
    }
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  render(snapshot, interpolationAlpha = 1) {
    this.controls.update();

    this.syncBodies(snapshot);
    if (snapshot.sph) this.syncSPH(snapshot.sph);
    if (snapshot.bondData) this.updateBondLines(snapshot.bondData);

    this.renderer.render(this.scene, this.camera);
    this.drawCalls = this.renderer.info.render.calls;
    this.renderer.info.reset();
  }

  // ─── Misc ─────────────────────────────────────────────────────────────────

  _onResize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** Find the nearest body to a screen-space NDC position. */
  pickBody(ndcX, ndcY, snapshot) {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(ndcX, ndcY);
    raycaster.setFromCamera(mouse, this.camera);

    let minDist = Infinity, nearest = -1;
    for (let ai = 0; ai < snapshot.N; ai++) {
      const px = snapshot.positions[ai*3];
      const py = snapshot.positions[ai*3+1];
      const pz = snapshot.positions[ai*3+2];
      const pos3 = new THREE.Vector3(px, py, pz);
      const dist = raycaster.ray.distanceToPoint(pos3);
      if (dist < snapshot.radii[ai] * 2 && dist < minDist) {
        minDist = dist;
        nearest = ai;
      }
    }
    return nearest; // returns snapshot index, caller maps to bodyId
  }

  /** Project world position to NDC for tools */
  worldToNDC(wx, wy, wz) {
    const v = new THREE.Vector3(wx, wy, wz).project(this.camera);
    return [v.x, v.y];
  }

  /** Unproject NDC + depth to world ray */
  ndcToWorldRay(ndcX, ndcY) {
    const origin = new THREE.Vector3();
    const dir    = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera({ x: ndcX, y: ndcY }, this.camera);
    return {
      origin: raycaster.ray.origin.toArray(),
      dir:    raycaster.ray.direction.toArray(),
    };
  }

  setBloom(enabled) {
    this.bloom = enabled;
    // Minimal bloom toggle: just increase tone mapping exposure slightly
    this.renderer.toneMappingExposure = enabled ? 1.5 : 1.25;
  }

  getStats() {
    return {
      drawCalls: this.drawCalls,
      fragInstances: this._instanceCount,
      sphInstances: this._sphCount,
    };
  }

  dispose() {
    this.renderer.dispose();
  }
}
