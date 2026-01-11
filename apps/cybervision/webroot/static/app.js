/* CyberVision - Main application */

import { initGPU } from "./gpu.js";

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
    this.gpuContext = null;
    this.stream = null;
    this.isRunning = false;
    this.animationFrame = null;

    // Effect state
    this.currentEffect = "halftone";
    this.dotSize = 8;

    // WebGPU resources
    this.halftoneShader = null;
    this.halftonePipeline = null;
    this.halftoneBindGroup = null;
    this.uniformBuffer = null;
    this.inputTexture = null;
    this.outputTexture = null;

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    this.init();
  }

  async init() {
    // Check WebGPU support
    try {
      this.gpuContext = await initGPU(this.canvas);
      this.gpuStatus.textContent = "Supported";
      this.gpuStatus.style.color = "#34d399";
      await this.initHalftoneEffect();
    } catch (err) {
      this.gpuStatus.textContent = "Not supported";
      this.gpuStatus.style.color = "#f87171";
      this.setStatus(`WebGPU Error: ${err.message}`);
      this.startBtn.disabled = true;
      return;
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
      this.updateHalftoneParams();
    });

    this.setStatus("Ready. Click 'Start Camera' to begin.");
  }

  async initHalftoneEffect() {
    // Load shader
    const shaderResponse = await fetch("/static/shaders/halftone.wgsl");
    const shaderCode = await shaderResponse.text();
    this.halftoneShader = this.gpuContext.device.createShaderModule({
      code: shaderCode,
    });

    // Create uniform buffer
    const uniformData = new Float32Array([
      this.dotSize, // sampleSize
      0,            // width (will be updated)
      0,            // height (will be updated)
      0,            // padding
    ]);
    this.uniformBuffer = this.gpuContext.createUniformBuffer(uniformData);
  }

  updateHalftoneParams() {
    if (!this.uniformBuffer || !this.video.videoWidth) return;

    const uniformData = new Float32Array([
      this.dotSize,
      this.video.videoWidth,
      this.video.videoHeight,
      0,
    ]);
    this.gpuContext.updateUniformBuffer(this.uniformBuffer, uniformData);
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

      // Reconfigure canvas context
      this.gpuContext.canvasContext.configure({
        device: this.gpuContext.device,
        format: this.gpuContext.canvasFormat,
        alphaMode: "opaque",
        width: this.canvas.width,
        height: this.canvas.height,
      });

      // Setup halftone pipeline
      console.log("Setting up halftone pipeline...");
      await this.setupHalftonePipeline();

      // Update params with video dimensions
      this.updateHalftoneParams();
      console.log("Halftone pipeline ready");

      // Update UI
      this.startBtn.disabled = true;
      this.stopBtn.disabled = false;
      this.isRunning = true;

      this.setStatus("Camera running.");

      // Start render loop
      this.render();
    } catch (err) {
      this.setStatus(`Camera Error: ${err.message}`);
      console.error(err);
    }
  }

  async setupHalftonePipeline() {
    const width = this.video.videoWidth;
    const height = this.video.videoHeight;

    // Create textures
    this.inputTexture = this.gpuContext.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Output texture for compute shader (must be rgba8unorm for storage)
    this.outputTexture = this.gpuContext.device.createTexture({
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Create compute pipeline
    this.halftonePipeline = this.gpuContext.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.halftoneShader,
        entryPoint: "main",
      },
    });

    // Create bind group
    this.halftoneBindGroup = this.gpuContext.device.createBindGroup({
      layout: this.halftonePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
      ],
    });

    // Create blit pipeline to copy rgba texture to bgra canvas
    await this.setupBlitPipeline();
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

    const blitShader = this.gpuContext.device.createShaderModule({
      code: blitShaderCode,
    });

    this.blitPipeline = this.gpuContext.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: blitShader,
        entryPoint: "vs_main",
      },
      fragment: {
        module: blitShader,
        entryPoint: "fs_main",
        targets: [{ format: this.gpuContext.canvasFormat }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    // Create sampler
    this.blitSampler = this.gpuContext.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Create bind group
    this.blitBindGroup = this.gpuContext.device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.blitSampler,
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
      // Copy video frame to input texture
      this.gpuContext.device.queue.copyExternalImageToTexture(
        { source: this.video, flipY: false },
        { texture: this.inputTexture },
        [this.video.videoWidth, this.video.videoHeight]
      );

      if (this.currentEffect === "halftone") {
        this.renderHalftone();
      } else {
        this.renderPassthrough();
      }

      // Update FPS
      this.updateFPS();
    } catch (err) {
      console.error("Render error:", err);
      this.setStatus(`Render Error: ${err.message}`);
      this.stopCamera();
      return;
    }

    // Continue loop
    this.animationFrame = requestAnimationFrame(() => this.render());
  }

  renderHalftone() {
    const commandEncoder = this.gpuContext.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.halftonePipeline);
    computePass.setBindGroup(0, this.halftoneBindGroup);

    const workgroupsX = Math.ceil(this.video.videoWidth / 8);
    const workgroupsY = Math.ceil(this.video.videoHeight / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    // Blit rgba texture to canvas (bgra format)
    const canvasTexture = this.gpuContext.canvasContext.getCurrentTexture();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, this.blitBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    this.gpuContext.device.queue.submit([commandEncoder.finish()]);
  }

  renderPassthrough() {
    const commandEncoder = this.gpuContext.device.createCommandEncoder();

    // Create temporary bind group for input texture
    const passthroughBindGroup = this.gpuContext.device.createBindGroup({
      layout: this.blitPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.blitSampler,
        },
      ],
    });

    // Blit input texture directly to canvas
    const canvasTexture = this.gpuContext.canvasContext.getCurrentTexture();
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, passthroughBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    this.gpuContext.device.queue.submit([commandEncoder.finish()]);
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
