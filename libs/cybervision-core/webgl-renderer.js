/* WebGL renderer for CyberVision (fallback for browsers without WebGPU) */

export class WebGLRenderer {
  constructor() {
    this.gl = null;
    this.canvas = null;
    this.halftoneProgram = null;
    this.clusteringProgram = null;
    this.edgesProgram = null;
    this.mosaicProgram = null;
    this.duotoneProgram = null;
    this.ditherProgram = null;
    this.posterizeProgram = null;
    this.twirlProgram = null;
    this.vignetteProgram = null;
    this.chromaticProgram = null;
    this.glitchProgram = null;
    this.thermalProgram = null;
    this.kaleidoscopeProgram = null;
    this.passthroughProgram = null;
    this.videoTexture = null;
    this.positionBuffer = null;
    this.texCoordBuffer = null;
    // Cached shader locations for performance
    this.halftoneLocations = null;
    this.clusteringLocations = null;
    this.edgesLocations = null;
    this.mosaicLocations = null;
    this.duotoneLocations = null;
    this.ditherLocations = null;
    this.posterizeLocations = null;
    this.twirlLocations = null;
    this.vignetteLocations = null;
    this.chromaticLocations = null;
    this.glitchLocations = null;
    this.thermalLocations = null;
    this.kaleidoscopeLocations = null;
    this.passthroughLocations = null;
  }

  async init(canvas) {
    this.canvas = canvas;

    // Get WebGL2 context
    this.gl = canvas.getContext("webgl2");
    if (!this.gl) {
      throw new Error("WebGL2 not supported");
    }

    console.log("WebGL2 initialized");

    // Create shaders
    await this.createShaders();

    // Create buffers for fullscreen quad
    this.createBuffers();

    // Create video texture
    this.videoTexture = this.createTexture();

    return this;
  }

  async loadShader(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Failed to load shader: ${path}`);
    }
    return await response.text();
  }

  async createShaders() {
    const gl = this.gl;

    // Load shaders from files
    const vertexShaderSource = await this.loadShader("/shaders/common.vert.glsl");
    const halftoneFragmentSource = await this.loadShader("/shaders/halftone.frag.glsl");
    const clusteringFragmentSource = await this.loadShader("/shaders/clustering.frag.glsl");
    const edgesFragmentSource = await this.loadShader("/shaders/edges.frag.glsl");
    const mosaicFragmentSource = await this.loadShader("/shaders/mosaic.frag.glsl");
    const duotoneFragmentSource = await this.loadShader("/shaders/duotone.frag.glsl");
    const ditherFragmentSource = await this.loadShader("/shaders/dither.frag.glsl");
    const posterizeFragmentSource = await this.loadShader("/shaders/posterize.frag.glsl");
    const twirlFragmentSource = await this.loadShader("/shaders/twirl.frag.glsl");
    const vignetteFragmentSource = await this.loadShader("/shaders/vignette.frag.glsl");
    const chromaticFragmentSource = await this.loadShader("/shaders/chromatic.frag.glsl");
    const glitchFragmentSource = await this.loadShader("/shaders/glitch.frag.glsl");
    const thermalFragmentSource = await this.loadShader("/shaders/thermal.frag.glsl");
    const kaleidoscopeFragmentSource = await this.loadShader("/shaders/kaleidoscope.frag.glsl");
    const passthroughFragmentSource = await this.loadShader("/shaders/passthrough.frag.glsl");

    // Compile shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const halftoneFragment = this.compileShader(gl.FRAGMENT_SHADER, halftoneFragmentSource);
    const clusteringFragment = this.compileShader(gl.FRAGMENT_SHADER, clusteringFragmentSource);
    const edgesFragment = this.compileShader(gl.FRAGMENT_SHADER, edgesFragmentSource);
    const mosaicFragment = this.compileShader(gl.FRAGMENT_SHADER, mosaicFragmentSource);
    const duotoneFragment = this.compileShader(gl.FRAGMENT_SHADER, duotoneFragmentSource);
    const ditherFragment = this.compileShader(gl.FRAGMENT_SHADER, ditherFragmentSource);
    const posterizeFragment = this.compileShader(gl.FRAGMENT_SHADER, posterizeFragmentSource);
    const twirlFragment = this.compileShader(gl.FRAGMENT_SHADER, twirlFragmentSource);
    const vignetteFragment = this.compileShader(gl.FRAGMENT_SHADER, vignetteFragmentSource);
    const chromaticFragment = this.compileShader(gl.FRAGMENT_SHADER, chromaticFragmentSource);
    const glitchFragment = this.compileShader(gl.FRAGMENT_SHADER, glitchFragmentSource);
    const thermalFragment = this.compileShader(gl.FRAGMENT_SHADER, thermalFragmentSource);
    const kaleidoscopeFragment = this.compileShader(gl.FRAGMENT_SHADER, kaleidoscopeFragmentSource);
    const passthroughFragment = this.compileShader(gl.FRAGMENT_SHADER, passthroughFragmentSource);

    // Create programs
    this.halftoneProgram = this.createProgram(vertexShader, halftoneFragment);
    this.clusteringProgram = this.createProgram(vertexShader, clusteringFragment);
    this.edgesProgram = this.createProgram(vertexShader, edgesFragment);
    this.mosaicProgram = this.createProgram(vertexShader, mosaicFragment);
    this.duotoneProgram = this.createProgram(vertexShader, duotoneFragment);
    this.ditherProgram = this.createProgram(vertexShader, ditherFragment);
    this.posterizeProgram = this.createProgram(vertexShader, posterizeFragment);
    this.twirlProgram = this.createProgram(vertexShader, twirlFragment);
    this.vignetteProgram = this.createProgram(vertexShader, vignetteFragment);
    this.chromaticProgram = this.createProgram(vertexShader, chromaticFragment);
    this.glitchProgram = this.createProgram(vertexShader, glitchFragment);
    this.thermalProgram = this.createProgram(vertexShader, thermalFragment);
    this.kaleidoscopeProgram = this.createProgram(vertexShader, kaleidoscopeFragment);
    this.passthroughProgram = this.createProgram(vertexShader, passthroughFragment);

    console.log("WebGL shaders compiled");
  }

  compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error("Shader compilation failed: " + info);
    }

    return shader;
  }

  createProgram(vertexShader, fragmentShader) {
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error("Program linking failed: " + info);
    }

    return program;
  }

  createBuffers() {
    const gl = this.gl;

    // Fullscreen quad vertices
    const positions = new Float32Array([
      -1, -1,  // bottom left
       1, -1,  // bottom right
      -1,  1,  // top left
       1,  1,  // top right
    ]);

    // Texture coordinates
    const texCoords = new Float32Array([
      0, 1,  // bottom left
      1, 1,  // bottom right
      0, 0,  // top left
      1, 0,  // top right
    ]);

    // Create position buffer
    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    // Create texcoord buffer
    this.texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, texCoords, gl.STATIC_DRAW);
  }

  createTexture() {
    const gl = this.gl;
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return texture;
  }

  updateVideoTexture(video) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      video
    );
  }

  renderHalftone(video, dotSize, useRandomColors = false) {
    const gl = this.gl;
    const program = this.halftoneProgram;

    // Cache locations on first render
    if (!this.halftoneLocations) {
      this.halftoneLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        dotSize: gl.getUniformLocation(program, "u_dotSize"),
        useRandomColors: gl.getUniformLocation(program, "u_useRandomColors"),
        time: gl.getUniformLocation(program, "u_time"),
      };
    }
    const locations = this.halftoneLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    const time = Math.floor(performance.now() / 1000);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.dotSize, dotSize);
    gl.uniform1f(locations.useRandomColors, useRandomColors ? 1.0 : 0.0);
    gl.uniform1f(locations.time, time);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderPassthrough(video) {
    const gl = this.gl;
    const program = this.passthroughProgram;

    // Cache locations on first render
    if (!this.passthroughLocations) {
      this.passthroughLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
      };
    }
    const locations = this.passthroughLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderClustering(video, algorithm, colorCount, threshold) {
    const gl = this.gl;
    const program = this.clusteringProgram;

    // Map algorithm string to number
    const algorithmMap = {
      "quantization-kmeans": 0,
      "quantization-kmeans-true": 1,
      "meanshift": 2,
      "meanshift-true": 3,
      "posterize": 4,
      "posterize-true": 5,
    };

    // Cache locations on first render
    if (!this.clusteringLocations) {
      this.clusteringLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        algorithm: gl.getUniformLocation(program, "u_algorithm"),
        colorCount: gl.getUniformLocation(program, "u_colorCount"),
        threshold: gl.getUniformLocation(program, "u_threshold"),
      };
    }
    const locations = this.clusteringLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.algorithm, algorithmMap[algorithm] || 0);
    gl.uniform1f(locations.colorCount, colorCount);
    gl.uniform1f(locations.threshold, threshold);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderEdges(video, algorithm, threshold, showOverlay, invert, edgeColor, thickness) {
    const gl = this.gl;
    const program = this.edgesProgram;

    // Map algorithm string to number
    const algorithmMap = {
      "sobel": 0,
      "prewitt": 1,
      "laplacian": 2,
      "canny": 3,
    };

    // Cache locations on first render
    if (!this.edgesLocations) {
      this.edgesLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        algorithm: gl.getUniformLocation(program, "u_algorithm"),
        threshold: gl.getUniformLocation(program, "u_threshold"),
        showOverlay: gl.getUniformLocation(program, "u_showOverlay"),
        invert: gl.getUniformLocation(program, "u_invert"),
        edgeColor: gl.getUniformLocation(program, "u_edgeColor"),
        thickness: gl.getUniformLocation(program, "u_thickness"),
      };
    }
    const locations = this.edgesLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.algorithm, algorithmMap[algorithm] || 0);
    gl.uniform1f(locations.threshold, threshold);
    gl.uniform1f(locations.showOverlay, showOverlay ? 1.0 : 0.0);
    gl.uniform1f(locations.invert, invert ? 1.0 : 0.0);
    gl.uniform3f(locations.edgeColor, edgeColor[0], edgeColor[1], edgeColor[2]);
    gl.uniform1f(locations.thickness, thickness);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderMosaic(video, blockSize, mode) {
    const gl = this.gl;
    const program = this.mosaicProgram;

    // Map mode string to number
    const modeMap = {
      "center": 0,
      "average": 1,
      "min": 2,
      "max": 3,
      "dominant": 4,
      "random": 5,
    };

    // Cache locations on first render
    if (!this.mosaicLocations) {
      this.mosaicLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        blockSize: gl.getUniformLocation(program, "u_blockSize"),
        mode: gl.getUniformLocation(program, "u_mode"),
        time: gl.getUniformLocation(program, "u_time"),
      };
    }
    const locations = this.mosaicLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    const time = Math.floor(performance.now() / 1000);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.blockSize, blockSize);
    gl.uniform1f(locations.mode, modeMap[mode] || 0);
    gl.uniform1f(locations.time, time);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderDuotone(video, shadowColor, highlightColor) {
    const gl = this.gl;
    const program = this.duotoneProgram;

    if (!this.duotoneLocations) {
      this.duotoneLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        shadowColor: gl.getUniformLocation(program, "u_shadowColor"),
        highlightColor: gl.getUniformLocation(program, "u_highlightColor"),
      };
    }
    const locations = this.duotoneLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform3f(locations.shadowColor, shadowColor[0], shadowColor[1], shadowColor[2]);
    gl.uniform3f(locations.highlightColor, highlightColor[0], highlightColor[1], highlightColor[2]);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderDither(video, scale, levels) {
    const gl = this.gl;
    const program = this.ditherProgram;

    if (!this.ditherLocations) {
      this.ditherLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        scale: gl.getUniformLocation(program, "u_scale"),
        levels: gl.getUniformLocation(program, "u_levels"),
      };
    }
    const locations = this.ditherLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.scale, scale);
    gl.uniform1f(locations.levels, levels);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderPosterize(video, levels) {
    const gl = this.gl;
    const program = this.posterizeProgram;

    if (!this.posterizeLocations) {
      this.posterizeLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        levels: gl.getUniformLocation(program, "u_levels"),
      };
    }
    const locations = this.posterizeLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform1f(locations.levels, levels);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderTwirl(video, centerX, centerY, radius, strength) {
    const gl = this.gl;
    const program = this.twirlProgram;

    if (!this.twirlLocations) {
      this.twirlLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        center: gl.getUniformLocation(program, "u_center"),
        radius: gl.getUniformLocation(program, "u_radius"),
        strength: gl.getUniformLocation(program, "u_strength"),
      };
    }
    const locations = this.twirlLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.center, centerX, centerY);
    gl.uniform1f(locations.radius, radius);
    gl.uniform1f(locations.strength, strength);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderVignette(video, vignetteAmount, grainAmount) {
    const gl = this.gl;
    const program = this.vignetteProgram;

    if (!this.vignetteLocations) {
      this.vignetteLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        vignette: gl.getUniformLocation(program, "u_vignette"),
        grain: gl.getUniformLocation(program, "u_grain"),
        time: gl.getUniformLocation(program, "u_time"),
      };
    }
    const locations = this.vignetteLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    const time = performance.now() / 1000;

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.vignette, vignetteAmount);
    gl.uniform1f(locations.grain, grainAmount);
    gl.uniform1f(locations.time, time);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderChromatic(video, intensity, mode, centerX, centerY) {
    const gl = this.gl;
    const program = this.chromaticProgram;

    // Map mode string to number
    const modeMap = {
      "radial": 0,
      "horizontal": 1,
      "vertical": 2,
    };

    // Cache locations on first render
    if (!this.chromaticLocations) {
      this.chromaticLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        intensity: gl.getUniformLocation(program, "u_intensity"),
        mode: gl.getUniformLocation(program, "u_mode"),
        centerX: gl.getUniformLocation(program, "u_centerX"),
        centerY: gl.getUniformLocation(program, "u_centerY"),
      };
    }
    const locations = this.chromaticLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.intensity, intensity);
    gl.uniform1f(locations.mode, modeMap[mode] || 0);
    gl.uniform1f(locations.centerX, centerX);
    gl.uniform1f(locations.centerY, centerY);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderGlitch(video, mode, intensity, blockSize, colorShift, noiseAmount, scanlineStrength) {
    const gl = this.gl;
    const program = this.glitchProgram;

    const modeMap = {
      "slices": 0,
      "blocks": 1,
      "scanlines": 2,
    };

    // Cache locations on first render
    if (!this.glitchLocations) {
      this.glitchLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        mode: gl.getUniformLocation(program, "u_mode"),
        intensity: gl.getUniformLocation(program, "u_intensity"),
        blockSize: gl.getUniformLocation(program, "u_blockSize"),
        colorShift: gl.getUniformLocation(program, "u_colorShift"),
        noiseAmount: gl.getUniformLocation(program, "u_noiseAmount"),
        scanlineStrength: gl.getUniformLocation(program, "u_scanlineStrength"),
        time: gl.getUniformLocation(program, "u_time"),
      };
    }
    const locations = this.glitchLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.mode, modeMap[mode] || 0);
    gl.uniform1f(locations.intensity, intensity);
    gl.uniform1f(locations.blockSize, blockSize);
    gl.uniform1f(locations.colorShift, colorShift);
    gl.uniform1f(locations.noiseAmount, noiseAmount);
    gl.uniform1f(locations.scanlineStrength, scanlineStrength);
    gl.uniform1f(locations.time, performance.now() / 1000);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderThermal(video, palette, contrast, invert) {
    const gl = this.gl;
    const program = this.thermalProgram;

    // Map palette string to number
    const paletteMap = {
      "classic": 0,
      "infrared": 1,
      "fire": 2,
    };

    // Cache locations on first render
    if (!this.thermalLocations) {
      this.thermalLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        resolution: gl.getUniformLocation(program, "u_resolution"),
        palette: gl.getUniformLocation(program, "u_palette"),
        contrast: gl.getUniformLocation(program, "u_contrast"),
        invert: gl.getUniformLocation(program, "u_invert"),
      };
    }
    const locations = this.thermalLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    gl.uniform1i(locations.video, 0);
    gl.uniform2f(locations.resolution, video.videoWidth, video.videoHeight);
    gl.uniform1f(locations.palette, paletteMap[palette] || 0);
    gl.uniform1f(locations.contrast, contrast);
    gl.uniform1f(locations.invert, invert ? 1.0 : 0.0);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderKaleidoscope(video, segments, rotationSpeed) {
    const gl = this.gl;
    const program = this.kaleidoscopeProgram;

    // Cache locations on first render
    if (!this.kaleidoscopeLocations) {
      this.kaleidoscopeLocations = {
        position: gl.getAttribLocation(program, "a_position"),
        texCoord: gl.getAttribLocation(program, "a_texCoord"),
        video: gl.getUniformLocation(program, "u_video"),
        segments: gl.getUniformLocation(program, "u_segments"),
        rotationSpeed: gl.getUniformLocation(program, "u_rotationSpeed"),
        time: gl.getUniformLocation(program, "u_time"),
      };
    }
    const locations = this.kaleidoscopeLocations;

    this.updateVideoTexture(video);

    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(locations.position);
    gl.vertexAttribPointer(locations.position, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(locations.texCoord);
    gl.vertexAttribPointer(locations.texCoord, 2, gl.FLOAT, false, 0, 0);

    const time = performance.now() / 1000;

    gl.uniform1i(locations.video, 0);
    gl.uniform1f(locations.segments, segments);
    gl.uniform1f(locations.rotationSpeed, rotationSpeed);
    gl.uniform1f(locations.time, time);

    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  destroy() {
    const gl = this.gl;
    if (gl) {
      if (this.videoTexture) gl.deleteTexture(this.videoTexture);
      if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
      if (this.texCoordBuffer) gl.deleteBuffer(this.texCoordBuffer);
      if (this.halftoneProgram) gl.deleteProgram(this.halftoneProgram);
      if (this.edgesProgram) gl.deleteProgram(this.edgesProgram);
      if (this.mosaicProgram) gl.deleteProgram(this.mosaicProgram);
      if (this.chromaticProgram) gl.deleteProgram(this.chromaticProgram);
      if (this.glitchProgram) gl.deleteProgram(this.glitchProgram);
      if (this.thermalProgram) gl.deleteProgram(this.thermalProgram);
      if (this.passthroughProgram) gl.deleteProgram(this.passthroughProgram);
    }
  }
}

export async function initWebGL(canvas) {
  const renderer = new WebGLRenderer();
  await renderer.init(canvas);
  return renderer;
}
