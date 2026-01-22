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
    this.maskTexture = null;  // For segmentation mask
    this.backgroundTexture = null;  // For background replacement

    // Pipelines
    this.halftonePipeline = null;
    this.clusteringPipeline = null;
    this.edgesPipeline = null;
    this.mosaicPipeline = null;
    this.duotonePipeline = null;
    this.ditherPipeline = null;
    this.twirlPipeline = null;
    this.chromaticPipeline = null;
    this.glitchPipeline = null;
    this.thermalPipeline = null;
    this.pixelSortSegmentPipeline = null;
    this.pixelSortPipeline = null;
    this.rotatePipeline = null;
    this.kaleidoscopePipeline = null;
    this.segmentationPipeline = null;
    this.blitPipeline = null;

    // Bind groups
    this.halftoneBindGroup = null;
    this.clusteringBindGroup = null;
    this.edgesBindGroup = null;
    this.mosaicBindGroup = null;
    this.duotoneBindGroup = null;
    this.ditherBindGroup = null;
    this.twirlBindGroup = null;
    this.chromaticBindGroup = null;
    this.glitchBindGroup = null;
    this.thermalBindGroup = null;
    this.pixelSortSegmentBindGroup = null;
    this.pixelSortBindGroup = null;
    this.rotateBindGroup = null;
    this.rotateBackBindGroup = null;
    this.kaleidoscopeBindGroup = null;
    this.segmentationBindGroup = null;
    this.blitBindGroup = null;
    this.passthroughBindGroup = null;

    // Buffers and samplers
    this.uniformBuffer = null;
    this.clusteringUniformBuffer = null;
    this.edgesUniformBuffer = null;
    this.mosaicUniformBuffer = null;
    this.duotoneUniformBuffer = null;
    this.ditherUniformBuffer = null;
    this.twirlUniformBuffer = null;
    this.chromaticUniformBuffer = null;
    this.glitchUniformBuffer = null;
    this.thermalUniformBuffer = null;
    this.pixelSortSegmentUniformBuffer = null;
    this.pixelSortUniformBuffer = null;
    this.rotateUniformBuffer = null;
    this.kaleidoscopeUniformBuffer = null;
    this.segmentationUniformBuffer = null;
    this.blitSampler = null;

    // Shader modules
    this.halftoneShader = null;
    this.clusteringShader = null;
    this.edgesShader = null;
    this.mosaicShader = null;
    this.duotoneShader = null;
    this.ditherShader = null;
    this.twirlShader = null;
    this.chromaticShader = null;
    this.glitchShader = null;
    this.thermalShader = null;
    this.pixelSortSegmentShader = null;
    this.pixelSortShader = null;
    this.rotateShader = null;
    this.kaleidoscopeShader = null;
    this.segmentationShader = null;

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
    const shaderResponse = await fetch("/shaders/halftone.wgsl");
    const shaderCode = await shaderResponse.text();
    this.halftoneShader = this.device.createShaderModule({
      code: shaderCode,
    });

    // Load and create clustering shader
    const clusteringShaderResponse = await fetch("/shaders/clustering.wgsl");
    const clusteringShaderCode = await clusteringShaderResponse.text();
    this.clusteringShader = this.device.createShaderModule({
      code: clusteringShaderCode,
    });

    // Load and create edges shader
    const edgesShaderResponse = await fetch("/shaders/edges.wgsl");
    const edgesShaderCode = await edgesShaderResponse.text();
    this.edgesShader = this.device.createShaderModule({
      code: edgesShaderCode,
    });

    // Load and create mosaic shader
    const mosaicShaderResponse = await fetch("/shaders/mosaic.wgsl");
    const mosaicShaderCode = await mosaicShaderResponse.text();
    this.mosaicShader = this.device.createShaderModule({
      code: mosaicShaderCode,
    });

    // Load and create duotone shader
    const duotoneShaderResponse = await fetch("/shaders/duotone.wgsl");
    const duotoneShaderCode = await duotoneShaderResponse.text();
    this.duotoneShader = this.device.createShaderModule({
      code: duotoneShaderCode,
    });

    // Load and create dither shader
    const ditherShaderResponse = await fetch("/shaders/dither.wgsl");
    const ditherShaderCode = await ditherShaderResponse.text();
    this.ditherShader = this.device.createShaderModule({
      code: ditherShaderCode,
    });

    // Load and create twirl shader
    const twirlShaderResponse = await fetch("/shaders/twirl.wgsl");
    const twirlShaderCode = await twirlShaderResponse.text();
    this.twirlShader = this.device.createShaderModule({
      code: twirlShaderCode,
    });

    // Load and create chromatic shader
    const chromaticShaderResponse = await fetch("/shaders/chromatic.wgsl");
    const chromaticShaderCode = await chromaticShaderResponse.text();
    this.chromaticShader = this.device.createShaderModule({
      code: chromaticShaderCode,
    });

    // Load and create thermal shader
    const thermalShaderResponse = await fetch("/shaders/thermal.wgsl");
    const thermalShaderCode = await thermalShaderResponse.text();
    this.thermalShader = this.device.createShaderModule({
      code: thermalShaderCode,
    });

    // Load and create glitch shader
    const glitchShaderResponse = await fetch("/shaders/glitch.wgsl");
    const glitchShaderCode = await glitchShaderResponse.text();
    this.glitchShader = this.device.createShaderModule({
      code: glitchShaderCode,
    });

    // Load and create pixel sort segment shader
    const pixelSortSegmentShaderResponse = await fetch("/shaders/pixelsort-segment.wgsl");
    const pixelSortSegmentShaderCode = await pixelSortSegmentShaderResponse.text();
    this.pixelSortSegmentShader = this.device.createShaderModule({
      code: pixelSortSegmentShaderCode,
    });

    // Load and create pixel sort shader
    const pixelSortShaderResponse = await fetch("/shaders/pixelsort.wgsl");
    const pixelSortShaderCode = await pixelSortShaderResponse.text();
    this.pixelSortShader = this.device.createShaderModule({
      code: pixelSortShaderCode,
    });

    // Load and create rotate shader
    const rotateShaderResponse = await fetch("/shaders/rotate.wgsl");
    const rotateShaderCode = await rotateShaderResponse.text();
    this.rotateShader = this.device.createShaderModule({
      code: rotateShaderCode,
    });

    // Load and create kaleidoscope shader
    const kaleidoscopeShaderResponse = await fetch("/shaders/kaleidoscope.wgsl");
    const kaleidoscopeShaderCode = await kaleidoscopeShaderResponse.text();
    this.kaleidoscopeShader = this.device.createShaderModule({
      code: kaleidoscopeShaderCode,
    });

    // Load and create segmentation shader
    const segmentationShaderResponse = await fetch("/shaders/segmentation.wgsl");
    const segmentationShaderCode = await segmentationShaderResponse.text();
    this.segmentationShader = this.device.createShaderModule({
      code: segmentationShaderCode,
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

    // Create duotone uniform buffer
    const duotoneUniformData = new Float32Array([
      0.0, 0.0, 0.0, 0.0,  // shadow color
      1.0, 1.0, 1.0, 0.0,  // highlight color
    ]);
    this.duotoneUniformBuffer = this.createUniformBuffer(duotoneUniformData);

    // Create dither uniform buffer
    const ditherUniformData = new Float32Array([
      2.0,  // scale
      4.0,  // levels
      0.0,  // padding
      0.0,  // padding
    ]);
    this.ditherUniformBuffer = this.createUniformBuffer(ditherUniformData);

    // Create twirl uniform buffer
    const twirlUniformData = new Float32Array([
      0.5,  // centerX
      0.5,  // centerY
      0.5,  // radius (normalized)
      0.0,  // strength
    ]);
    this.twirlUniformBuffer = this.createUniformBuffer(twirlUniformData);

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
    ]);
    this.kaleidoscopeUniformBuffer = this.createUniformBuffer(kaleidoscopeUniformData);

    // Create segmentation uniform buffer
    const segmentationUniformData = new Uint32Array([
      0,  // mode (0=blur, 1=replace, 2=blackout)
      0,  // padding
    ]);
    const segmentationFloatData = new Float32Array([
      10.0,  // blurRadius
      0.5,   // threshold
      0.05,  // feather
      0,     // padding
    ]);
    const segmentationDimensionsData = new Uint32Array([
      this.videoWidth,
      this.videoHeight,
    ]);

    // Combine all data into one buffer (struct layout: u32 mode, f32 blurRadius, f32 threshold, f32 feather, u32 width, u32 height, u32 softEdges, u32 glow)
    const segmentationBufferData = new ArrayBuffer(32);  // 8 * 4 bytes = 32 bytes
    const segmentationView = new DataView(segmentationBufferData);
    segmentationView.setUint32(0, segmentationUniformData[0], true);  // mode
    segmentationView.setFloat32(4, segmentationFloatData[0], true);   // blurRadius
    segmentationView.setFloat32(8, segmentationFloatData[1], true);   // threshold
    segmentationView.setFloat32(12, segmentationFloatData[2], true);  // feather
    segmentationView.setUint32(16, segmentationDimensionsData[0], true);  // width
    segmentationView.setUint32(20, segmentationDimensionsData[1], true);  // height
    segmentationView.setUint32(24, 1, true);  // softEdges (default on)
    segmentationView.setUint32(28, 0, true);  // glow (default off)

    this.segmentationUniformBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.segmentationUniformBuffer, 0, segmentationBufferData);

    // Create reusable buffer for uniform updates (avoids per-frame allocations)
    this.segmentationUniformsArrayBuffer = new ArrayBuffer(32);
    this.segmentationUniformsU32 = new Uint32Array(this.segmentationUniformsArrayBuffer);
    this.segmentationUniformsF32 = new Float32Array(this.segmentationUniformsArrayBuffer);

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

    // Create mask texture for segmentation (256x256, will be upsampled in shader)
    this.maskTexture = this.device.createTexture({
      size: [256, 256],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Create background texture for replacement mode (same size as video)
    this.backgroundTexture = this.device.createTexture({
      size: [this.videoWidth, this.videoHeight],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
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

    // Create duotone compute pipeline
    this.duotonePipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.duotoneShader,
        entryPoint: "main",
      },
    });

    // Create duotone bind group
    this.duotoneBindGroup = this.device.createBindGroup({
      layout: this.duotonePipeline.getBindGroupLayout(0),
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
            buffer: this.duotoneUniformBuffer,
          },
        },
      ],
    });

    // Create dither compute pipeline
    this.ditherPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.ditherShader,
        entryPoint: "main",
      },
    });

    // Create dither bind group
    this.ditherBindGroup = this.device.createBindGroup({
      layout: this.ditherPipeline.getBindGroupLayout(0),
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
            buffer: this.ditherUniformBuffer,
          },
        },
      ],
    });

    // Create twirl compute pipeline
    this.twirlPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.twirlShader,
        entryPoint: "main",
      },
    });

    // Create twirl bind group
    this.twirlBindGroup = this.device.createBindGroup({
      layout: this.twirlPipeline.getBindGroupLayout(0),
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
            buffer: this.twirlUniformBuffer,
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

    // Create segmentation compute pipeline
    this.segmentationPipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.segmentationShader,
        entryPoint: "main",
      },
    });

    // Create segmentation bind group
    this.segmentationBindGroup = this.device.createBindGroup({
      layout: this.segmentationPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.inputTexture.createView(),
        },
        {
          binding: 1,
          resource: this.maskTexture.createView(),
        },
        {
          binding: 2,
          resource: this.backgroundTexture.createView(),
        },
        {
          binding: 3,
          resource: this.outputTexture.createView(),
        },
        {
          binding: 4,
          resource: {
            buffer: this.segmentationUniformBuffer,
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
    this.uploadVideoToTexture(video);

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
    this.uploadVideoToTexture(video);

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

  renderClustering(video, algorithm, colorCount, threshold) {
    // Map algorithm string to number
    const algorithmMap = {
      "quantization-kmeans": 0,
      "quantization-kmeans-true": 1,
      "meanshift": 2,
      "meanshift-true": 3,
      "posterize": 4,
      "posterize-true": 5,
    };

    // Update uniform buffer with current parameters
    const uniformData = new Float32Array([
      algorithmMap[algorithm] || 0,
      colorCount,
      threshold,
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.clusteringUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.clusteringPipeline);
    computePass.setBindGroup(0, this.clusteringBindGroup);

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

  renderEdges(video, algorithm, threshold, showOverlay, invert, edgeColor, thickness) {
    // Map algorithm string to number
    const algorithmMap = {
      "sobel": 0,
      "prewitt": 1,
      "laplacian": 2,
      "canny": 3,
    };

    // Update uniform buffer with current parameters
    const uniformData = new Float32Array([
      algorithmMap[algorithm] || 0,
      threshold,
      showOverlay ? 1.0 : 0.0,
      invert ? 1.0 : 0.0,
      edgeColor[0], edgeColor[1], edgeColor[2],
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      thickness,
      0,  // padding
    ]);
    this.updateUniformBuffer(this.edgesUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.edgesPipeline);
    computePass.setBindGroup(0, this.edgesBindGroup);

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

  renderMosaic(video, blockSize, mode) {
    // Map mode string to number
    const modeMap = {
      "center": 0,
      "average": 1,
      "min": 2,
      "max": 3,
      "dominant": 4,
      "random": 5,
    };

    // Update uniform buffer with current parameters
    const time = Math.floor(performance.now() / 1000);
    const uniformData = new Float32Array([
      blockSize,
      modeMap[mode] || 0,
      this.videoWidth,
      this.videoHeight,
      time,
      0, 0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.mosaicUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();

    if (mode === "dominant") {
      // Per-block dispatch for dominant mode (uses shared histogram)
      computePass.setPipeline(this.mosaicDominantPipeline);
      computePass.setBindGroup(0, this.mosaicDominantBindGroup);

      const blocksX = Math.ceil(this.videoWidth / blockSize);
      const blocksY = Math.ceil(this.videoHeight / blockSize);
      computePass.dispatchWorkgroups(blocksX, blocksY);
    } else {
      // Per-pixel dispatch for other modes
      computePass.setPipeline(this.mosaicPipeline);
      computePass.setBindGroup(0, this.mosaicBindGroup);

      const workgroupsX = Math.ceil(this.videoWidth / 8);
      const workgroupsY = Math.ceil(this.videoHeight / 8);
      computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    }

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

  renderDuotone(video, shadowColor, highlightColor) {
    const uniformData = new Float32Array([
      shadowColor[0], shadowColor[1], shadowColor[2], 0.0,
      highlightColor[0], highlightColor[1], highlightColor[2], 0.0,
    ]);
    this.updateUniformBuffer(this.duotoneUniformBuffer, uniformData);

    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.duotonePipeline);
    computePass.setBindGroup(0, this.duotoneBindGroup);

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

  renderDither(video, scale, levels) {
    const uniformData = new Float32Array([
      scale,
      levels,
      0.0,
      0.0,
    ]);
    this.updateUniformBuffer(this.ditherUniformBuffer, uniformData);

    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.ditherPipeline);
    computePass.setBindGroup(0, this.ditherBindGroup);

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

  renderTwirl(video, centerX, centerY, radius, strength) {
    const uniformData = new Float32Array([
      centerX,
      centerY,
      radius,
      strength,
    ]);
    this.updateUniformBuffer(this.twirlUniformBuffer, uniformData);

    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.twirlPipeline);
    computePass.setBindGroup(0, this.twirlBindGroup);

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

  renderChromatic(video, intensity, mode, centerX, centerY) {
    // Map mode string to number
    const modeMap = {
      "radial": 0,
      "horizontal": 1,
      "vertical": 2,
    };

    // Update uniform buffer with current parameters
    const uniformData = new Float32Array([
      intensity,
      modeMap[mode] || 0,
      centerX,
      centerY,
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.chromaticUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.chromaticPipeline);
    computePass.setBindGroup(0, this.chromaticBindGroup);

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

  renderGlitch(video, mode, intensity, blockSize, colorShift, noiseAmount, scanlineStrength) {
    // Map mode string to number
    const modeMap = {
      "slices": 0,
      "blocks": 1,
      "scanlines": 2,
    };

    const uniformData = new Float32Array([
      modeMap[mode] || 0,
      intensity,
      blockSize,
      colorShift,
      noiseAmount,
      scanlineStrength,
      performance.now() / 1000,
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.glitchUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.glitchPipeline);
    computePass.setBindGroup(0, this.glitchBindGroup);

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

  renderThermal(video, palette, contrast, invert) {
    // Map palette string to number
    const paletteMap = {
      "classic": 0,
      "infrared": 1,
      "fire": 2,
    };

    // Update uniform buffer with current parameters
    const uniformData = new Float32Array([
      paletteMap[palette] || 0,
      contrast,
      invert ? 1.0 : 0.0,
      0,  // padding
      this.videoWidth,
      this.videoHeight,
      0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.thermalUniformBuffer, uniformData);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.thermalPipeline);
    computePass.setBindGroup(0, this.thermalBindGroup);

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

  renderPixelSort(video, angleMode, direction, angle, thresholdLow, thresholdHigh, thresholdMode, sortKey, sortOrder, algorithm, iterations) {
    // Map strings to numbers
    const directionMap = {
      "horizontal": 0,
      "vertical": 1,
      "diagonal_right": 2,
      "diagonal_left": 3,
    };

    const thresholdModeMap = {
      "brightness": 0,
      "saturation": 1,
      "hue": 2,
      "edge": 3,
    };

    const sortKeyMap = {
      "luminance": 0,
      "hue": 1,
      "saturation": 2,
      "red": 3,
      "green": 4,
      "blue": 5,
    };

    const algorithmMap = {
      "bitonic": 0,
      "bubble": 1,
    };

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const workgroupsX = Math.ceil(this.videoWidth / 8);
    const workgroupsY = Math.ceil(this.videoHeight / 8);

    // Determine if we need to use rotation (custom angle or diagonal presets)
    const useRotation = angleMode === "rotate" || direction === "diagonal_right" || direction === "diagonal_left";
    let rotationAngle = angle;

    // Map diagonal presets to angles for rotate-sort-rotate approach
    if (direction === "diagonal_right") {
      rotationAngle = 45;  // ↘ diagonal
    } else if (direction === "diagonal_left") {
      rotationAngle = -45;  // ↙ diagonal (or 135°)
    }

    // If using rotation (custom angle or diagonal), rotate the image first
    if (useRotation) {
      const angleRadians = -(rotationAngle * Math.PI) / 180.0;  // Negative because we rotate the sort direction
      const rotateUniformData = new Float32Array([
        angleRadians,
        this.videoWidth,
        this.videoHeight,
        0,
      ]);
      this.updateUniformBuffer(this.rotateUniformBuffer, rotateUniformData);

      const rotateEncoder = this.device.createCommandEncoder();
      const rotatePass = rotateEncoder.beginComputePass();
      rotatePass.setPipeline(this.rotatePipeline);
      rotatePass.setBindGroup(0, this.rotateBindGroup);
      rotatePass.dispatchWorkgroups(workgroupsX, workgroupsY);
      rotatePass.end();
      this.device.queue.submit([rotateEncoder.finish()]);

      // Now temp texture has rotated image
      // Next: segment identification on temp → output
    }

    // Pass 1: Segment identification
    const segmentUniformData = new Float32Array([
      thresholdLow,
      thresholdHigh,
      this.videoWidth,
      this.videoHeight,
      thresholdModeMap[thresholdMode] || 0,
      0, 0, 0,  // padding
    ]);
    this.updateUniformBuffer(this.pixelSortSegmentUniformBuffer, segmentUniformData);

    const commandEncoder = this.device.createCommandEncoder();

    let computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.pixelSortSegmentPipeline);
    // Use appropriate bind group based on whether we rotated
    if (useRotation) {
      computePass.setBindGroup(0, this.pixelSortSegmentBindGroupRotate);  // temp -> output
    } else {
      computePass.setBindGroup(0, this.pixelSortSegmentBindGroup);  // input -> output
    }
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Pass 2+: Sorting passes with ping-pong between output and temp
    const algo = algorithmMap[algorithm] || 0;
    // If we used rotation (custom angle or diagonals), always use horizontal sorting
    // Otherwise use the specified direction (horizontal or vertical only now)
    const sortDirection = useRotation ? 0 : (directionMap[direction] || 0);

    let passCount = 0;

    if (algo === 0) {
      // Bitonic sort - multiple passes
      // Use full dimension based on sort direction for complete sorting
      const maxSegmentLength = (sortDirection === 0) ? this.videoWidth : this.videoHeight;
      const numStages = Math.ceil(Math.log2(maxSegmentLength));

      for (let stage = 0; stage < numStages; stage++) {
        for (let step = stage; step >= 0; step--) {
          const sortUniformData = new Float32Array([
            thresholdLow,
            thresholdHigh,
            this.videoWidth,
            this.videoHeight,
            sortKeyMap[sortKey] || 0,
            sortOrder === "ascending" ? 0.0 : 1.0,
            sortDirection,
            algo,
            stage,
            step,
            0,  // iteration (unused for bitonic)
            0,  // padding
          ]);
          this.updateUniformBuffer(this.pixelSortUniformBuffer, sortUniformData);

          const sortCommandEncoder = this.device.createCommandEncoder();
          const sortComputePass = sortCommandEncoder.beginComputePass();
          sortComputePass.setPipeline(this.pixelSortPipeline);
          // Ping-pong: even passes output->temp, odd passes temp->output
          if (passCount % 2 === 0) {
            sortComputePass.setBindGroup(0, this.pixelSortBindGroupA);  // output -> temp
          } else {
            sortComputePass.setBindGroup(0, this.pixelSortBindGroupB);  // temp -> output
          }
          sortComputePass.dispatchWorkgroups(workgroupsX, workgroupsY);
          sortComputePass.end();

          this.device.queue.submit([sortCommandEncoder.finish()]);
          passCount++;
        }
      }
    } else {
      // Bubble sort - iterations passes
      for (let iter = 0; iter < iterations; iter++) {
        const sortUniformData = new Float32Array([
          thresholdLow,
          thresholdHigh,
          this.videoWidth,
          this.videoHeight,
          sortKeyMap[sortKey] || 0,
          sortOrder === "ascending" ? 0.0 : 1.0,
          sortDirection,
          algo,
          0,  // stage (unused for bubble)
          0,  // step (unused for bubble)
          iter,
          0,  // padding
        ]);
        this.updateUniformBuffer(this.pixelSortUniformBuffer, sortUniformData);

        const sortCommandEncoder = this.device.createCommandEncoder();
        const sortComputePass = sortCommandEncoder.beginComputePass();
        sortComputePass.setPipeline(this.pixelSortPipeline);
        // Ping-pong: even passes output->temp, odd passes temp->output
        if (passCount % 2 === 0) {
          sortComputePass.setBindGroup(0, this.pixelSortBindGroupA);  // output -> temp
        } else {
          sortComputePass.setBindGroup(0, this.pixelSortBindGroupB);  // temp -> output
        }
        sortComputePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        sortComputePass.end();

        this.device.queue.submit([sortCommandEncoder.finish()]);
        passCount++;
      }
    }

    // After sorting, the final result is in temp if odd passes, output if even passes
    const finalInTemp = passCount % 2 === 1;

    // If we used rotation (custom angle or diagonals), rotate back
    if (useRotation) {
      const angleRadians = (rotationAngle * Math.PI) / 180.0;  // Positive to rotate back
      const rotateBackUniformData = new Float32Array([
        angleRadians,
        this.videoWidth,
        this.videoHeight,
        0,
      ]);
      this.updateUniformBuffer(this.rotateUniformBuffer, rotateBackUniformData);

      const rotateBackEncoder = this.device.createCommandEncoder();
      const rotateBackPass = rotateBackEncoder.beginComputePass();
      rotateBackPass.setPipeline(this.rotatePipeline);

      // Use appropriate bind group based on where the sorted result is
      if (finalInTemp) {
        // Sorted result is in temp, rotate temp -> output directly
        rotateBackPass.setBindGroup(0, this.rotateTempToOutputBindGroup);
      } else {
        // Sorted result is in output, rotate output -> temp, then copy back
        rotateBackPass.setBindGroup(0, this.rotateBackBindGroup);
      }

      rotateBackPass.dispatchWorkgroups(workgroupsX, workgroupsY);
      rotateBackPass.end();
      this.device.queue.submit([rotateBackEncoder.finish()]);

      // If we rotated output->temp, copy temp back to output for blit
      if (!finalInTemp) {
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyTextureToTexture(
          { texture: this.tempTexture },
          { texture: this.outputTexture },
          [this.videoWidth, this.videoHeight]
        );
        this.device.queue.submit([copyEncoder.finish()]);
      }
      // else: result is already in output, ready for blit
    } else {
      // For preset mode, ensure final result is in outputTexture
      if (finalInTemp) {
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyTextureToTexture(
          { texture: this.tempTexture },
          { texture: this.outputTexture },
          [this.videoWidth, this.videoHeight]
        );
        this.device.queue.submit([copyEncoder.finish()]);
      }
    }

    // Final pass: Blit output to canvas
    const blitCommandEncoder = this.device.createCommandEncoder();
    const canvasTexture = this.canvasContext.getCurrentTexture();
    const renderPass = blitCommandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, this.blitBindGroup);  // Always blit from output
    renderPass.draw(6, 1, 0, 0);
    renderPass.end();

    this.device.queue.submit([blitCommandEncoder.finish()]);
  }

  renderKaleidoscope(video, segments, rotationSpeed) {
    const time = performance.now() / 1000;

    const uniformData = new Float32Array([
      segments,
      rotationSpeed,
      time,
      0.0,
    ]);
    this.updateUniformBuffer(this.kaleidoscopeUniformBuffer, uniformData);

    this.uploadVideoToTexture(video);

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

  /**
   * Update segmentation mask texture from ML inference result
   * @param {Uint8Array} maskData - Binary mask data (256x256)
   */
  updateSegmentationMask(maskData) {
    // Write single-channel mask data directly to r8unorm texture
    this.device.queue.writeTexture(
      { texture: this.maskTexture },
      maskData,
      { bytesPerRow: 256, rowsPerImage: 256 },
      { width: 256, height: 256 }
    );
  }

  /**
   * Update background texture for replacement mode
   * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} image - Background image
   */
  updateBackgroundImage(image) {
    // Create temporary canvas to resize/fit image
    const canvas = document.createElement('canvas');
    canvas.width = this.videoWidth;
    canvas.height = this.videoHeight;
    const ctx = canvas.getContext('2d');

    // Draw image to fit (cover mode)
    const imgAspect = image.width / image.height;
    const canvasAspect = this.videoWidth / this.videoHeight;

    let drawWidth, drawHeight, drawX, drawY;

    if (imgAspect > canvasAspect) {
      // Image is wider
      drawHeight = this.videoHeight;
      drawWidth = image.width * (this.videoHeight / image.height);
      drawX = (this.videoWidth - drawWidth) / 2;
      drawY = 0;
    } else {
      // Image is taller or same aspect
      drawWidth = this.videoWidth;
      drawHeight = image.height * (this.videoWidth / image.width);
      drawX = 0;
      drawY = (this.videoHeight - drawHeight) / 2;
    }

    ctx.drawImage(image, drawX, drawY, drawWidth, drawHeight);

    // Copy canvas to GPU texture
    this.device.queue.copyExternalImageToTexture(
      { source: canvas, flipY: false },
      { texture: this.backgroundTexture },
      [this.videoWidth, this.videoHeight]
    );
  }

  /**
   * Render segmentation effect
   * @param {HTMLVideoElement} video - Video element
   * @param {string} mode - Effect mode: "blur", "replace", "blackout"
   * @param {number} blurRadius - Blur radius for blur mode
   * @param {Uint8Array} maskData - Segmentation mask (256x256)
   * @param {boolean} softEdges - Enable soft edges
   * @param {boolean} glow - Enable glow effect
   */
  renderSegmentation(video, mode, blurRadius, maskData, softEdges, glow) {
    // Map mode string to number
    const modeMap = {
      "blur": 0,
      "replace": 1,
      "blackout": 2,
    };

    // Update mask texture if provided
    if (maskData) {
      this.updateSegmentationMask(maskData);
    }

    // Update uniform buffer with current parameters (reuse buffer to avoid allocations)
    this.segmentationUniformsU32[0] = modeMap[mode] || 0;  // mode
    this.segmentationUniformsF32[1] = blurRadius;          // blurRadius
    this.segmentationUniformsF32[2] = 0.5;                 // threshold
    this.segmentationUniformsF32[3] = 0.05;                // feather
    this.segmentationUniformsU32[4] = this.videoWidth;     // width
    this.segmentationUniformsU32[5] = this.videoHeight;    // height
    this.segmentationUniformsU32[6] = softEdges ? 1 : 0;   // softEdges
    this.segmentationUniformsU32[7] = glow ? 1 : 0;        // glow

    this.device.queue.writeBuffer(this.segmentationUniformBuffer, 0, this.segmentationUniformsArrayBuffer);

    // Copy video frame to input texture
    this.uploadVideoToTexture(video);

    const commandEncoder = this.device.createCommandEncoder();

    // Run compute shader
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.segmentationPipeline);
    computePass.setBindGroup(0, this.segmentationBindGroup);

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

  uploadVideoToTexture(video) {
    if (!video) return;
    const srcWidth = video.videoWidth || video.width || this.videoWidth;
    const srcHeight = video.videoHeight || video.height || this.videoHeight;
    const width = Math.min(this.videoWidth, srcWidth);
    const height = Math.min(this.videoHeight, srcHeight);
    
    if (width <= 0 || height <= 0) return;

    this.device.queue.copyExternalImageToTexture(
      { source: video, flipY: false },
      { texture: this.inputTexture },
      [width, height]
    );
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
