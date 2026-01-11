/* WebGPU renderer for CyberVision */

export class WebGPURenderer {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.canvas = null;
    this.canvasContext = null;
    this.canvasFormat = null;

    // Textures
    this.inputTexture = null;
    this.outputTexture = null;

    // Pipelines
    this.halftonePipeline = null;
    this.blitPipeline = null;

    // Bind groups
    this.halftoneBindGroup = null;
    this.blitBindGroup = null;
    this.passthroughBindGroup = null;

    // Buffers and samplers
    this.uniformBuffer = null;
    this.blitSampler = null;

    // Shader modules
    this.halftoneShader = null;

    // Video dimensions and effect params
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.dotSize = 8;
    this.coloredDots = 3;
  }

  async init(canvas) {
    this.canvas = canvas;

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

    console.log("WebGPU initialized");
    return this;
  }

  async setupPipeline(video, dotSize, coloredDots = 3) {
    this.videoWidth = video.videoWidth;
    this.videoHeight = video.videoHeight;
    this.dotSize = dotSize;
    this.coloredDots = coloredDots;

    // Reconfigure canvas context with video dimensions
    this.canvasContext.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: "opaque",
      width: this.videoWidth,
      height: this.videoHeight,
    });

    console.log("Setting up WebGPU pipeline...");

    // Load and create halftone shader
    const shaderResponse = await fetch("/static/shaders/halftone.wgsl");
    const shaderCode = await shaderResponse.text();
    this.halftoneShader = this.device.createShaderModule({
      code: shaderCode,
    });

    // Create uniform buffer
    const time = Math.floor(performance.now() / 1000);
    const uniformData = new Float32Array([
      dotSize,
      this.videoWidth,
      this.videoHeight,
      coloredDots,
      time,
      0, 0, 0,  // padding
    ]);
    this.uniformBuffer = this.createUniformBuffer(uniformData);

    // Create textures
    this.inputTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.outputTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Create compute pipeline
    this.halftonePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.halftoneShader,
        entryPoint: "main",
      },
    });

    // Create halftone bind group
    this.halftoneBindGroup = this.device.createBindGroup({
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

    const blitShader = this.device.createShaderModule({
      code: blitShaderCode,
    });

    this.blitPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: blitShader,
        entryPoint: "vs_main",
      },
      fragment: {
        module: blitShader,
        entryPoint: "fs_main",
        targets: [{ format: this.canvasFormat }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    this.blitSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Create bind group for halftone output -> canvas
    this.blitBindGroup = this.device.createBindGroup({
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

    // Create bind group for input texture -> canvas (passthrough mode)
    this.passthroughBindGroup = this.device.createBindGroup({
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
  }

  updateDotSize(dotSize) {
    this.dotSize = dotSize;
    if (this.uniformBuffer && this.videoWidth) {
      const time = Math.floor(performance.now() / 1000);
      const uniformData = new Float32Array([
        dotSize,
        this.videoWidth,
        this.videoHeight,
        this.coloredDots,
        time,
        0, 0, 0,  // padding
      ]);
      this.updateUniformBuffer(this.uniformBuffer, uniformData);
    }
  }

  renderHalftone(video) {
    // Update time in uniform buffer every frame
    const time = Math.floor(performance.now() / 1000);
    const uniformData = new Float32Array([
      this.dotSize,
      this.videoWidth,
      this.videoHeight,
      this.coloredDots,
      time,
      0, 0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.uniformBuffer, uniformData);

    // Copy video frame to input texture
    this.device.queue.copyExternalImageToTexture(
      { source: video, flipY: false },
      { texture: this.inputTexture },
      [this.videoWidth, this.videoHeight]
    );

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.halftonePipeline);
    computePass.setBindGroup(0, this.halftoneBindGroup);

    const workgroupsX = Math.ceil(this.videoWidth / 8);
    const workgroupsY = Math.ceil(this.videoHeight / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    // Blit rgba texture to canvas (bgra format)
    const canvasTexture = this.canvasContext.getCurrentTexture();
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

    this.device.queue.submit([commandEncoder.finish()]);
  }

  renderPassthrough(video) {
    // Copy video frame to input texture
    this.device.queue.copyExternalImageToTexture(
      { source: video, flipY: false },
      { texture: this.inputTexture },
      [this.videoWidth, this.videoHeight]
    );

    const commandEncoder = this.device.createCommandEncoder();

    // Blit input texture directly to canvas
    const canvasTexture = this.canvasContext.getCurrentTexture();
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
    renderPass.setBindGroup(0, this.passthroughBindGroup);
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
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
  const renderer = new WebGPURenderer();
  await renderer.init(canvas);
  return renderer;
}
