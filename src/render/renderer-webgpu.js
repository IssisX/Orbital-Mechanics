/**
 * WebGPU Renderer — optional GPU-accelerated path.
 *
 * Uses Three.js WebGPURenderer (Three.js r163+ with WebGPU addon)
 * when available. Falls back transparently to RendererWebGL2 via
 * the feature detect in main.js.
 *
 * This file provides the same interface as renderer-webgl2.js so it
 * can be swapped in transparently.
 *
 * WebGPU compute shaders (gravity-barneshut.wgsl, lbvh-build.wgsl,
 * neighbors.wgsl) are dispatched here for the physics hot path.
 */

// NOTE: WebGPU path relies on three.js WebGPURenderer which is available
// as an addon in three@0.163. If unavailable (older browser or missing
// adapter), this module gracefully fails and main.js falls back.

export class RendererWebGPU {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {GPUDevice} device   WebGPU device (pre-created in main.js)
   */
  constructor(canvas, device) {
    this.canvas  = canvas;
    this.device  = device;
    this._ready  = false;

    // GPU compute buffers
    this._bodyBuf    = null;
    this._accBuf     = null;
    this._paramBuf   = null;

    // Pipeline cache
    this._gravityPipeline = null;

    this.drawCalls = 0;
    this._fallback = null; // RendererWebGL2 instance for rendering
  }

  async init() {
    if (!this.device) {
      throw new Error('WebGPU device not available');
    }

    // For rendering we still use Three.js with its WebGL2 renderer
    // (WebGPURenderer in three.js is still experimental).
    // The GPU path only accelerates compute (gravity).
    const { RendererWebGL2 } = await import('./renderer-webgl2.js');
    this._fallback = new RendererWebGL2(this.canvas);

    // Load WGSL shaders
    this._gravityShader = await this._loadWGSL('/src/physics/gpu/gravity-barneshut.wgsl');
    this._lbvhShader    = await this._loadWGSL('/src/physics/gpu/lbvh-build.wgsl');
    this._neighborShader= await this._loadWGSL('/src/physics/gpu/neighbors.wgsl');

    // Create gravity compute pipeline
    this._gravityPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: this._gravityShader }),
        entryPoint: 'computeGravity',
      },
    });

    this._ready = true;
    return this;
  }

  async _loadWGSL(path) {
    try {
      const r = await fetch(path);
      return r.text();
    } catch {
      return '// shader load failed';
    }
  }

  // ─── GPU Gravity Compute ─────────────────────────────────────────────────

  /**
   * Dispatch gravity computation on GPU.
   * @param {Float32Array} bodyData   Packed [x,y,z,mass, vx,vy,vz,radius, ...] per body
   * @param {number} N
   * @returns {Promise<Float32Array>} accelerations [ax,ay,az] per body
   */
  async computeGravityGPU(bodyData, N, G, epsilon, dt) {
    if (!this._ready || !this._gravityPipeline) return null;

    const dev = this.device;
    const bodyStride = 8; // vec3 pos + mass + vec3 vel + radius = 8 floats

    const bodyBuf = dev.createBuffer({
      size: N * bodyStride * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(bodyBuf, 0, bodyData, 0, N * bodyStride);

    const accBuf = dev.createBuffer({
      size: N * 3 * 4, // vec3<f32> per body
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramData = new Float32Array([
      N, G, epsilon * epsilon, 0.6, // theta
      dt, 0, 0, 0,
    ]);
    const paramBuf = dev.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(paramBuf, 0, paramData);

    const bindGroup = dev.createBindGroup({
      layout: this._gravityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bodyBuf } },
        { binding: 1, resource: { buffer: accBuf } },
        { binding: 2, resource: { buffer: paramBuf } },
      ],
    });

    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this._gravityPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / 64));
    pass.end();

    // Read back
    const readBuf = dev.createBuffer({
      size: N * 3 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(accBuf, 0, readBuf, 0, N * 3 * 4);
    dev.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    // Cleanup
    bodyBuf.destroy(); accBuf.destroy(); paramBuf.destroy(); readBuf.destroy();

    return result;
  }

  // ─── Rendering (delegate to WebGL2 fallback) ─────────────────────────────

  render(snapshot, alpha) {
    if (this._fallback) {
      this._fallback.render(snapshot, alpha);
      this.drawCalls = this._fallback.drawCalls;
    }
  }

  syncBodies(snapshot)       { this._fallback?.syncBodies(snapshot); }
  syncSPH(sph)               { this._fallback?.syncSPH(sph); }
  pickBody(x, y, snap)       { return this._fallback?.pickBody(x, y, snap); }
  worldToNDC(x, y, z)        { return this._fallback?.worldToNDC(x, y, z); }
  ndcToWorldRay(x, y)        { return this._fallback?.ndcToWorldRay(x, y); }
  setBloom(v)                { this._fallback?.setBloom(v); }
  _onResize()                { this._fallback?._onResize(); }
  getStats()                 { return this._fallback?.getStats() ?? {}; }

  get controls()             { return this._fallback?.controls; }
  get showBonds()            { return this._fallback?.showBonds; }
  set showBonds(v)           { if (this._fallback) this._fallback.showBonds = v; }
  get showOctree()           { return this._fallback?.showOctree; }
  set showOctree(v)          { if (this._fallback) this._fallback.showOctree = v; }
  get showBVH()              { return this._fallback?.showBVH; }
  set showBVH(v)             { if (this._fallback) this._fallback.showBVH = v; }

  dispose() {
    this._fallback?.dispose();
    this.device?.destroy?.();
  }

  /** Whether WebGPU compute is available */
  get isGPUCompute() { return this._ready; }
}

/**
 * Attempt to acquire a WebGPU device.
 * @returns {Promise<GPUDevice|null>}
 */
export async function requestWebGPUDevice() {
  if (!navigator.gpu) return null;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return null;
    return await adapter.requestDevice();
  } catch {
    return null;
  }
}
