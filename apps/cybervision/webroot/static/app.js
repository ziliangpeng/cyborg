/* CyberVision - Main application */

import { initGPU } from "./webgpu-renderer.js";
import { initWebGL } from "./webgl-renderer.js";

class CyberVision {
  constructor() {
    // DOM elements
    this.video = document.getElementById("video");
    this.canvas = document.getElementById("canvas");
    this.startBtn = document.getElementById("startBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.statusEl = document.getElementById("status");
    this.effectSelect = document.getElementById("effectSelect");
    this.dotSizeSlider = document.getElementById("dotSizeSlider");
    this.dotSizeValue = document.getElementById("dotSizeValue");
    this.fpsValue = document.getElementById("fpsValue");
    this.gpuStatus = document.getElementById("gpuStatus");

    // State
    this.renderer = null;
    this.rendererType = null; // 'webgpu' or 'webgl'
    this.stream = null;
    this.isRunning = false;
    this.animationFrame = null;

    // Effect state
    this.currentEffect = "halftone";
    this.dotSize = 8;

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    this.init();
  }

  async init() {
    // Try WebGPU first, fallback to WebGL
    try {
      console.log("Attempting to initialize WebGPU...");
      this.renderer = await initGPU(this.canvas);
      this.rendererType = "webgpu";
      this.gpuStatus.textContent = "WebGPU";
      this.gpuStatus.style.color = "#34d399";
      console.log("WebGPU initialized successfully");
    } catch (err) {
      console.log("WebGPU failed, trying WebGL:", err.message);
      try {
        this.renderer = await initWebGL(this.canvas);
        this.rendererType = "webgl";
        this.gpuStatus.textContent = "WebGL";
        this.gpuStatus.style.color = "#60a5fa";
        console.log("WebGL initialized successfully");
      } catch (err2) {
        this.gpuStatus.textContent = "Not supported";
        this.gpuStatus.style.color = "#f87171";
        this.setStatus(`Error: Neither WebGPU nor WebGL2 are supported. ${err2.message}`);
        this.startBtn.disabled = true;
        return;
      }
    }

    // Event listeners
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.stopBtn.addEventListener("click", () => this.stopCamera());
    this.effectSelect.addEventListener("change", (e) => {
      this.currentEffect = e.target.value;
    });
    this.dotSizeSlider.addEventListener("input", (e) => {
      this.dotSize = parseInt(e.target.value);
      this.dotSizeValue.textContent = this.dotSize;
      if (this.rendererType === "webgpu") {
        this.updateHalftoneParams();
      }
    });

    this.setStatus(`Ready. Using ${this.rendererType.toUpperCase()}. Click 'Start Camera' to begin.`);
  }

  updateHalftoneParams() {
    // Only for WebGPU
    if (this.rendererType === "webgpu" && this.renderer.uniformBuffer && this.video.videoWidth) {
      const uniformData = new Float32Array([
        this.dotSize,
        this.video.videoWidth,
        this.video.videoHeight,
        0,
      ]);
      this.renderer.updateUniformBuffer(this.renderer.uniformBuffer, uniformData);
    }
  }

  async startCamera() {
    try {
      this.setStatus("Requesting camera access...");

      // Request camera
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      });

      this.video.srcObject = this.stream;

      // Wait for video metadata and ensure it's playing
      await new Promise((resolve, reject) => {
        this.video.onloadedmetadata = async () => {
          try {
            await this.video.play();
            console.log("Video playing");
            resolve();
          } catch (err) {
            reject(err);
          }
        };
      });

      // Setup canvas size
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;

      console.log(`Video dimensions: ${this.video.videoWidth}x${this.video.videoHeight}`);

      // Initialize renderer-specific resources
      if (this.rendererType === "webgpu") {
        await this.setupWebGPU();
      }
      // WebGL doesn't need additional setup after init

      // Update UI
      this.startBtn.disabled = true;
      this.stopBtn.disabled = false;
      this.isRunning = true;

      this.setStatus(`Camera running with ${this.rendererType.toUpperCase()}.`);

      // Start render loop
      this.render();
    } catch (err) {
      this.setStatus(`Camera Error: ${err.message}`);
      console.error(err);
    }
  }

  async setupWebGPU() {
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;

    // Reconfigure canvas context
    this.renderer.canvasContext.configure({
      device: this.renderer.device,
      format: this.renderer.canvasFormat,
      alphaMode: "opaque",
      width: width,
      height: height,
    });

    console.log("Setting up WebGPU pipeline...");

    // Load and create halftone shader
    const shaderResponse = await fetch("/static/shaders/halftone.wgsl");
    const shaderCode = await shaderResponse.text();
    this.renderer.halftoneShader = this.renderer.device.createShaderModule({
      code: shaderCode,
    });

    // Create uniform buffer
    const uniformData = new Float32Array([
      this.dotSize,
      width,
      height,
      0,
    ]);
    this.renderer.uniformBuffer = this.renderer.createUniformBuffer(uniformData);

    // Create textures
    this.renderer.inputTexture = this.renderer.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.renderer.outputTexture = this.renderer.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Create compute pipeline
    this.renderer.halftonePipeline = this.renderer.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.renderer.halftoneShader,
        entryPoint: "main",
      },
    });

    // Create bind group
    this.renderer.halftoneBindGroup = this.renderer.device.createBindGroup({
      layout: this.renderer.halftonePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.renderer.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.renderer.outputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.renderer.uniformBuffer,
          },
        },
      ],
    });

    // Setup blit pipeline for rgba->bgra conversion
    await this.setupBlitPipeline();

    console.log("WebGPU pipeline ready");
  }

  async setupBlitPipeline() {
    const blitShaderCode = `
      @vertex
      fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
        var pos = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
          vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0)
        );
        return vec4f(pos[idx], 0.0, 1.0);
      }

      @group(0) @binding(0) var srcTex: texture_2d<f32>;
      @group(0) @binding(1) var srcSampler: sampler;

      @fragment
      fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let dims = textureDimensions(srcTex);
        let uv = pos.xy / vec2f(f32(dims.x), f32(dims.y));
        return textureSample(srcTex, srcSampler, uv);
      }
    `;

    const blitShader = this.renderer.device.createShaderModule({
      code: blitShaderCode,
    });

    this.renderer.blitPipeline = this.renderer.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: blitShader,
        entryPoint: "vs_main",
      },
      fragment: {
        module: blitShader,
        entryPoint: "fs_main",
        targets: [{ format: this.renderer.canvasFormat }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.renderer.blitSampler = this.renderer.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    this.renderer.blitBindGroup = this.renderer.device.createBindGroup({
      layout: this.renderer.blitPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.renderer.outputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.renderer.blitSampler,
        },
      ],
    });
  }

  stopCamera() {
    this.isRunning = false;

    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
    this.setStatus("Camera stopped.");
  }

  render() {
    if (!this.isRunning) return;

    try {
      if (this.currentEffect === "halftone") {
        this.renderHalftone();
      } else {
        this.renderPassthrough();
      }

      this.updateFPS();
    } catch (err) {
      console.error("Render error:", err);
      this.setStatus(`Render Error: ${err.message}`);
      this.stopCamera();
      return;
    }

    this.animationFrame = requestAnimationFrame(() => this.render());
  }

  renderHalftone() {
    if (this.rendererType === "webgpu") {
      this.renderHalftoneWebGPU();
    } else {
      this.renderer.renderHalftone(this.video, this.dotSize);
    }
  }

  renderPassthrough() {
    if (this.rendererType === "webgpu") {
      this.renderPassthroughWebGPU();
    } else {
      this.renderer.renderPassthrough(this.video);
    }
  }

  renderHalftoneWebGPU() {
    const device = this.renderer.device;

    // Copy video frame to input texture
    device.queue.copyExternalImageToTexture(
      { source: this.video, flipY: false },
      { texture: this.renderer.inputTexture },
      [this.video.videoWidth, this.video.videoHeight]
    );

    const commandEncoder = device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.renderer.halftonePipeline);
    computePass.setBindGroup(0, this.renderer.halftoneBindGroup);

    const workgroupsX = Math.ceil(this.video.videoWidth / 8);
    const workgroupsY = Math.ceil(this.video.videoHeight / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    // Blit rgba texture to canvas (bgra format)
    const canvasTexture = this.renderer.canvasContext.getCurrentTexture();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.renderer.blitPipeline);
    renderPass.setBindGroup(0, this.renderer.blitBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  renderPassthroughWebGPU() {
    const device = this.renderer.device;

    // Copy video frame to input texture
    device.queue.copyExternalImageToTexture(
      { source: this.video, flipY: false },
      { texture: this.renderer.inputTexture },
      [this.video.videoWidth, this.video.videoHeight]
    );

    const commandEncoder = device.createCommandEncoder();

    // Create temporary bind group for input texture
    const passthroughBindGroup = device.createBindGroup({
      layout: this.renderer.blitPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.renderer.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.renderer.blitSampler,
        },
      ],
    });

    // Blit input texture directly to canvas
    const canvasTexture = this.renderer.canvasContext.getCurrentTexture();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.renderer.blitPipeline);
    renderPass.setBindGroup(0, passthroughBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  updateFPS() {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastFpsTime;

    if (elapsed >= 1000) {
      const fps = Math.round((this.frameCount / elapsed) * 1000);
      this.fpsValue.textContent = fps;
      this.frameCount = 0;
      this.lastFpsTime = now;
    }
  }

  setStatus(text) {
    this.statusEl.textContent = text;
  }
}

// Initialize app when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => new CyberVision());
} else {
  new CyberVision();
}
