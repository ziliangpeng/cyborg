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
    this.tempTexture = null;  // For rotation intermediate results

    // Pipelines
    this.halftonePipeline = null;
    this.clusteringPipeline = null;
    this.edgesPipeline = null;
    this.mosaicPipeline = null;
    this.chromaticPipeline = null;
    this.glitchPipeline = null;
    this.thermalPipeline = null;
    this.pixelSortSegmentPipeline = null;
    this.pixelSortPipeline = null;
    this.rotatePipeline = null;
    this.blitPipeline = null;

    // Bind groups
    this.halftoneBindGroup = null;
    this.clusteringBindGroup = null;
    this.edgesBindGroup = null;
    this.mosaicBindGroup = null;
    this.chromaticBindGroup = null;
    this.glitchBindGroup = null;
    this.thermalBindGroup = null;
    this.pixelSortSegmentBindGroup = null;
    this.pixelSortBindGroup = null;
    this.rotateBindGroup = null;
    this.rotateBackBindGroup = null;
    this.blitBindGroup = null;
    this.passthroughBindGroup = null;

    // Buffers and samplers
    this.uniformBuffer = null;
    this.clusteringUniformBuffer = null;
    this.edgesUniformBuffer = null;
    this.mosaicUniformBuffer = null;
    this.chromaticUniformBuffer = null;
    this.glitchUniformBuffer = null;
    this.thermalUniformBuffer = null;
    this.pixelSortSegmentUniformBuffer = null;
    this.pixelSortUniformBuffer = null;
    this.rotateUniformBuffer = null;
    this.blitSampler = null;

    // Shader modules
    this.halftoneShader = null;
    this.clusteringShader = null;
    this.edgesShader = null;
    this.mosaicShader = null;
    this.chromaticShader = null;
    this.glitchShader = null;
    this.thermalShader = null;
    this.pixelSortSegmentShader = null;
    this.pixelSortShader = null;
    this.rotateShader = null;
    this.kaleidoscopeShader = null;

    // Video dimensions and effect params
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.dotSize = 8;
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

  async setupPipeline(video, dotSize) {
    this.videoWidth = video.videoWidth;
    this.videoHeight = video.videoHeight;
    this.dotSize = dotSize;

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

    // Load and create clustering shader
    const clusteringShaderResponse = await fetch("/static/shaders/clustering.wgsl");
    const clusteringShaderCode = await clusteringShaderResponse.text();
    this.clusteringShader = this.device.createShaderModule({
      code: clusteringShaderCode,
    });

    // Load and create edges shader
    const edgesShaderResponse = await fetch("/static/shaders/edges.wgsl");
    const edgesShaderCode = await edgesShaderResponse.text();
    this.edgesShader = this.device.createShaderModule({
      code: edgesShaderCode,
    });

    // Load and create mosaic shader
    const mosaicShaderResponse = await fetch("/static/shaders/mosaic.wgsl");
    const mosaicShaderCode = await mosaicShaderResponse.text();
    this.mosaicShader = this.device.createShaderModule({
      code: mosaicShaderCode,
    });

    // Load and create chromatic shader
    const chromaticShaderResponse = await fetch("/static/shaders/chromatic.wgsl");
    const chromaticShaderCode = await chromaticShaderResponse.text();
    this.chromaticShader = this.device.createShaderModule({
      code: chromaticShaderCode,
    });

    // Load and create thermal shader
    const thermalShaderResponse = await fetch("/static/shaders/thermal.wgsl");
    const thermalShaderCode = await thermalShaderResponse.text();
    this.thermalShader = this.device.createShaderModule({
      code: thermalShaderCode,
    });

    // Load and create glitch shader
    const glitchShaderResponse = await fetch("/static/shaders/glitch.wgsl");
    const glitchShaderCode = await glitchShaderResponse.text();
    this.glitchShader = this.device.createShaderModule({
      code: glitchShaderCode,
    });

    // Load and create pixel sort segment shader
    const pixelSortSegmentShaderResponse = await fetch("/static/shaders/pixelsort-segment.wgsl");
    const pixelSortSegmentShaderCode = await pixelSortSegmentShaderResponse.text();
    this.pixelSortSegmentShader = this.device.createShaderModule({
      code: pixelSortSegmentShaderCode,
    });

    // Load and create pixel sort shader
    const pixelSortShaderResponse = await fetch("/static/shaders/pixelsort.wgsl");
    const pixelSortShaderCode = await pixelSortShaderResponse.text();
    this.pixelSortShader = this.device.createShaderModule({
      code: pixelSortShaderCode,
    });

    // Load and create rotate shader
    const rotateShaderResponse = await fetch("/static/shaders/rotate.wgsl");
    const rotateShaderCode = await rotateShaderResponse.text();
    this.rotateShader = this.device.createShaderModule({
      code: rotateShaderCode,
    });

    // Load and create kaleidoscope shader
    const kaleidoscopeShaderResponse = await fetch("/static/shaders/kaleidoscope.wgsl");
    const kaleidoscopeShaderCode = await kaleidoscopeShaderResponse.text();
    this.kaleidoscopeShader = this.device.createShaderModule({
      code: kaleidoscopeShaderCode,
    });

    // Create uniform buffer
    const time = Math.floor(performance.now() / 1000);
    const uniformData = new Float32Array([
      dotSize,
      this.videoWidth,
      this.videoHeight,
      0,  // useRandomColors (will be updated in render)
      time,
      0, 0, 0,  // padding
    ]);
    this.uniformBuffer = this.createUniformBuffer(uniformData);

    // Create clustering uniform buffer
    const clusteringUniformData = new Float32Array([
      0,  // algorithm (0=quantization, 1=kmeans, 2=meanshift, 3=posterize)
      8,  // colorCount
      0.1,  // threshold
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.clusteringUniformBuffer = this.createUniformBuffer(clusteringUniformData);

    // Create edges uniform buffer
    const edgesUniformData = new Float32Array([
      0,  // algorithm (0=Sobel, 1=Prewitt, 2=Laplacian, 3=Canny)
      0.1,  // threshold
      0,  // showOverlay
      0,  // invert
      1.0, 1.0, 1.0,  // edgeColor RGB (white)
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      1,  // thickness
      0,  // padding
    ]);
    this.edgesUniformBuffer = this.createUniformBuffer(edgesUniformData);

    // Create mosaic uniform buffer
    const mosaicUniformData = new Float32Array([
      8,  // blockSize
      0,  // mode (0=Center, 1=Average, 2=Min, 3=Max, 4=Dominant, 5=Random)
      this.videoWidth,
      this.videoHeight,
      time,  // time for random mode
      0, 0, 0,  // padding
    ]);
    this.mosaicUniformBuffer = this.createUniformBuffer(mosaicUniformData);

    // Create chromatic uniform buffer
    const chromaticUniformData = new Float32Array([
      10,  // intensity
      0,  // mode (0=Radial, 1=Horizontal, 2=Vertical)
      0.5,  // centerX (0-1)
      0.5,  // centerY (0-1)
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.chromaticUniformBuffer = this.createUniformBuffer(chromaticUniformData);

    // Create glitch uniform buffer
    const glitchUniformData = new Float32Array([
      0,  // mode (0=Slices, 1=Blocks, 2=Scanlines)
      12,  // intensity
      24,  // blockSize
      4,  // colorShift
      0.15,  // noiseAmount
      0.3,  // scanlineStrength
      performance.now() / 1000,  // time
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.glitchUniformBuffer = this.createUniformBuffer(glitchUniformData);

    // Create thermal uniform buffer
    const thermalUniformData = new Float32Array([
      0,  // palette (0=Classic, 1=Infrared, 2=Fire)
      1.0,  // contrast
      0,  // invert
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.thermalUniformBuffer = this.createUniformBuffer(thermalUniformData);

    // Create pixel sort segment uniform buffer
    const pixelSortSegmentUniformData = new Float32Array([
      0.25,  // thresholdLow
      0.75,  // thresholdHigh
      this.videoWidth,
      this.videoHeight,
      0,  // thresholdMode (0=brightness)
      0, 0, 0,  // padding
    ]);
    this.pixelSortSegmentUniformBuffer = this.createUniformBuffer(pixelSortSegmentUniformData);

    // Create pixel sort uniform buffer
    const pixelSortUniformData = new Float32Array([
      0.25,  // thresholdLow
      0.75,  // thresholdHigh
      this.videoWidth,
      this.videoHeight,
      0,  // sortKey (0=luminance)
      0,  // sortOrder (0=ascending)
      0,  // direction (0=horizontal)
      0,  // algorithm (0=bitonic)
      0,  // stage (for bitonic)
      0,  // step (for bitonic)
      0,  // iteration (for bubble)
      0,  // padding
    ]);
    this.pixelSortUniformBuffer = this.createUniformBuffer(pixelSortUniformData);

    // Create rotate uniform buffer
    const rotateUniformData = new Float32Array([
      0.0,  // angle (will be updated)
      this.videoWidth,
      this.videoHeight,
      0,  // padding
    ]);
    this.rotateUniformBuffer = this.createUniformBuffer(rotateUniformData);

    // Create kaleidoscope uniform buffer
    const kaleidoscopeUniformData = new Float32Array([
      8.0,   // segments
      0.0,   // rotationSpeed
      0.0,   // time
      0.0,   // padding
      this.videoWidth,
      this.videoHeight,
      0.0, 0.0,  // padding
    ]);
    this.kaleidoscopeUniformBuffer = this.createUniformBuffer(kaleidoscopeUniformData);

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
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });

    // Create temporary texture for rotation intermediate results
    this.tempTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
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

    // Create clustering compute pipeline
    this.clusteringPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.clusteringShader,
        entryPoint: "main",
      },
    });

    // Create clustering bind group
    this.clusteringBindGroup = this.device.createBindGroup({
      layout: this.clusteringPipeline.getBindGroupLayout(0),
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
            buffer: this.clusteringUniformBuffer,
          },
        },
      ],
    });

    // Create edges compute pipeline
    this.edgesPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.edgesShader,
        entryPoint: "main",
      },
    });

    // Create edges bind group
    this.edgesBindGroup = this.device.createBindGroup({
      layout: this.edgesPipeline.getBindGroupLayout(0),
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
            buffer: this.edgesUniformBuffer,
          },
        },
      ],
    });

    // Create mosaic compute pipeline
    this.mosaicPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.mosaicShader,
        entryPoint: "main",
      },
    });

    // Create mosaic bind group
    this.mosaicBindGroup = this.device.createBindGroup({
      layout: this.mosaicPipeline.getBindGroupLayout(0),
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
            buffer: this.mosaicUniformBuffer,
          },
        },
      ],
    });

    // Create dominant mosaic compute pipeline (separate entry point for histogram-based mode)
    this.mosaicDominantPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.mosaicShader,
        entryPoint: "mainDominant",
      },
    });

    // Create dominant mosaic bind group (shares same bindings as regular mosaic)
    this.mosaicDominantBindGroup = this.device.createBindGroup({
      layout: this.mosaicDominantPipeline.getBindGroupLayout(0),
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
            buffer: this.mosaicUniformBuffer,
          },
        },
      ],
    });

    // Create chromatic compute pipeline
    this.chromaticPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.chromaticShader,
        entryPoint: "main",
      },
    });

    // Create chromatic bind group
    this.chromaticBindGroup = this.device.createBindGroup({
      layout: this.chromaticPipeline.getBindGroupLayout(0),
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
            buffer: this.chromaticUniformBuffer,
          },
        },
      ],
    });

    // Create glitch compute pipeline
    this.glitchPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.glitchShader,
        entryPoint: "main",
      },
    });

    // Create glitch bind group
    this.glitchBindGroup = this.device.createBindGroup({
      layout: this.glitchPipeline.getBindGroupLayout(0),
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
            buffer: this.glitchUniformBuffer,
          },
        },
      ],
    });

    // Create thermal compute pipeline
    this.thermalPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.thermalShader,
        entryPoint: "main",
      },
    });

    // Create thermal bind group
    this.thermalBindGroup = this.device.createBindGroup({
      layout: this.thermalPipeline.getBindGroupLayout(0),
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
            buffer: this.thermalUniformBuffer,
          },
        },
      ],
    });

    // Create pixel sort segment compute pipeline
    this.pixelSortSegmentPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.pixelSortSegmentShader,
        entryPoint: "main",
      },
    });

    // Create pixel sort segment bind group (input -> output)
    this.pixelSortSegmentBindGroup = this.device.createBindGroup({
      layout: this.pixelSortSegmentPipeline.getBindGroupLayout(0),
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
            buffer: this.pixelSortSegmentUniformBuffer,
          },
        },
      ],
    });

    // Create pixel sort segment bind group for rotate mode (temp -> output)
    this.pixelSortSegmentBindGroupRotate = this.device.createBindGroup({
      layout: this.pixelSortSegmentPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 1,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.pixelSortSegmentUniformBuffer,
          },
        },
      ],
    });

    // Create pixel sort compute pipeline
    this.pixelSortPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.pixelSortShader,
        entryPoint: "main",
      },
    });

    // Create pixel sort bind groups for ping-pong (output -> temp)
    this.pixelSortBindGroupA = this.device.createBindGroup({
      layout: this.pixelSortPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.pixelSortUniformBuffer,
          },
        },
      ],
    });

    // Create pixel sort bind groups for ping-pong (temp -> output)
    this.pixelSortBindGroupB = this.device.createBindGroup({
      layout: this.pixelSortPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 1,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.pixelSortUniformBuffer,
          },
        },
      ],
    });

    // Create rotate compute pipeline
    this.rotatePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.rotateShader,
        entryPoint: "main",
      },
    });

    // Create rotate bind group (input -> temp)
    this.rotateBindGroup = this.device.createBindGroup({
      layout: this.rotatePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.rotateUniformBuffer,
          },
        },
      ],
    });

    // Create rotate back bind group (output -> temp for final rotation)
    this.rotateBackBindGroup = this.device.createBindGroup({
      layout: this.rotatePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.rotateUniformBuffer,
          },
        },
      ],
    });

    // Create additional rotate bind group (temp -> output) to avoid copies
    this.rotateTempToOutputBindGroup = this.device.createBindGroup({
      layout: this.rotatePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.tempTexture.createView(),
        },
        {
          binding: 1,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: this.rotateUniformBuffer,
          },
        },
      ],
    });

    // Create kaleidoscope compute pipeline
    this.kaleidoscopePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.kaleidoscopeShader,
        entryPoint: "main",
      },
    });

    // Create kaleidoscope bind group
    this.kaleidoscopeBindGroup = this.device.createBindGroup({
      layout: this.kaleidoscopePipeline.getBindGroupLayout(0),
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
            buffer: this.kaleidoscopeUniformBuffer,
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
  }

  renderHalftone(video, useRandomColors = false) {
    // Update uniform buffer every frame with current time and color mode
    const time = Math.floor(performance.now() / 1000);
    const uniformData = new Float32Array([
      this.dotSize,
      this.videoWidth,
      this.videoHeight,
      useRandomColors ? 1.0 : 0.0,
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

  renderKaleidoscope(video, segments, rotationSpeed) {
    const time = performance.now() / 1000;

    const uniformData = new Float32Array([
      segments,
      rotationSpeed,
      time,
      0.0,
      this.videoWidth,
      this.videoHeight,
      0.0, 0.0,
    ]);
    this.updateUniformBuffer(this.kaleidoscopeUniformBuffer, uniformData);

    this.device.queue.copyExternalImageToTexture(
      { source: video, flipY: false },
      { texture: this.inputTexture },
      [this.videoWidth, this.videoHeight]
    );

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.kaleidoscopePipeline);
    computePass.setBindGroup(0, this.kaleidoscopeBindGroup);

    const workgroupsX = Math.ceil(this.videoWidth / 8);
    const workgroupsY = Math.ceil(this.videoHeight / 8);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

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
