/* WebGPU utilities for CyberVision */

export class GPUContext {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.canvasContext = null;
    this.canvasFormat = null;
  }

  async init(canvas) {
    // Check WebGPU support
    console.log("Checking WebGPU support...");
    console.log("navigator.gpu:", navigator.gpu);

    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser");
    }

    // Request adapter
    console.log("Requesting WebGPU adapter...");
    this.adapter = await navigator.gpu.requestAdapter();
    console.log("Adapter:", this.adapter);

    if (!this.adapter) {
      throw new Error("Failed to get WebGPU adapter");
    }

    // Request device
    console.log("Requesting WebGPU device...");
    this.device = await this.adapter.requestDevice();
    console.log("Device:", this.device);

    // Configure canvas
    this.canvasContext = canvas.getContext("webgpu");
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    this.canvasContext.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: "opaque",
    });

    return this;
  }

  createTextureFromVideo(video) {
    const texture = this.device.createTexture({
      size: [video.videoWidth, video.videoHeight],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Copy video frame to texture
    this.device.queue.copyExternalImageToTexture(
      { source: video, flipY: false },
      { texture },
      [video.videoWidth, video.videoHeight]
    );

    return texture;
  }

  createStorageTexture(width, height) {
    return this.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
  }

  createRenderTexture(width, height) {
    return this.device.createTexture({
      size: [width, height],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }

  async loadShader(url) {
    const response = await fetch(url);
    const code = await response.text();
    return this.device.createShaderModule({ code });
  }

  createUniformBuffer(data) {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    return buffer;
  }

  updateUniformBuffer(buffer, data) {
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  destroy() {
    if (this.device) {
      this.device.destroy();
    }
  }
}

export async function initGPU(canvas) {
  const ctx = new GPUContext();
  await ctx.init(canvas);
  return ctx;
}
