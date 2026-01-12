/* WebGL renderer for CyberVision (fallback for browsers without WebGPU) */

export class WebGLRenderer {
  constructor() {
    this.gl = null;
    this.canvas = null;
    this.halftoneProgram = null;
    this.clusteringProgram = null;
    this.edgesProgram = null;
    this.mosaicProgram = null;
    this.chromaticProgram = null;
    this.passthroughProgram = null;
    this.videoTexture = null;
    this.positionBuffer = null;
    this.texCoordBuffer = null;
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
    const vertexShaderSource = await this.loadShader("/static/shaders/common.vert.glsl");
    const halftoneFragmentSource = await this.loadShader("/static/shaders/halftone.frag.glsl");
    const clusteringFragmentSource = await this.loadShader("/static/shaders/clustering.frag.glsl");
    const edgesFragmentSource = await this.loadShader("/static/shaders/edges.frag.glsl");
    const mosaicFragmentSource = await this.loadShader("/static/shaders/mosaic.frag.glsl");
    const chromaticFragmentSource = await this.loadShader("/static/shaders/chromatic.frag.glsl");
    const passthroughFragmentSource = await this.loadShader("/static/shaders/passthrough.frag.glsl");

    /* OLD INLINE SHADERS - NOW LOADED FROM FILES
    // Vertex shader (same for all effects)
    const vertexShaderSource = `#version 300 es
      in vec2 a_position;
      in vec2 a_texCoord;
      out vec2 v_texCoord;

      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }
    `;

    // Halftone fragment shader
    const halftoneFragmentSource = `#version 300 es
      precision highp float;

      in vec2 v_texCoord;
      out vec4 fragColor;

      uniform sampler2D u_video;
      uniform vec2 u_resolution;
      uniform float u_dotSize;
      uniform float u_useRandomColors;
      uniform float u_time;

      // Simple hash function for pseudo-random number generation
      float hash(vec2 p) {
        vec3 p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
        float p3_dot = dot(p3, vec3(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
        vec3 p3_result = p3 + vec3(p3_dot);
        return fract((p3_result.x + p3_result.y) * p3_result.z);
      }

      // Generate random color based on cell position
      vec3 randomColor(vec2 cellIndex, float seed) {
        float seedOffset = seed * 127.1;  // Large multiplier breaks linear correlation
        float r = hash(cellIndex + vec2(seedOffset, seedOffset * 0.7));
        float g = hash(cellIndex + vec2(seedOffset * 0.3, seedOffset));
        float b = hash(cellIndex + vec2(seedOffset * 0.5, seedOffset * 0.9));
        return vec3(r, g, b);
      }

      void main() {
        vec2 pixelPos = v_texCoord * u_resolution;

        // Calculate which cell this pixel belongs to
        vec2 cellIndex = floor(pixelPos / u_dotSize);
        vec2 cellCenter = (cellIndex + 0.5) * u_dotSize;

        // Sample brightness at cell center
        vec2 centerUV = cellCenter / u_resolution;
        vec4 color = texture(u_video, centerUV);
        float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Calculate dot radius based on darkness
        float maxRadius = u_dotSize * 0.5;
        float radius = maxRadius * (1.0 - brightness);

        // Distance from pixel to cell center
        float dist = length(pixelPos - cellCenter);

        // Draw circle
        float inside = step(dist, radius);

        // Determine dot color based on useRandomColors flag
        vec3 dotColor = vec3(0.0);  // Default black

        if (u_useRandomColors > 0.5) {
          // Use random color that changes every second
          dotColor = randomColor(cellIndex, u_time);
        }

        // Colored/black dot on white background
        vec3 outputColor = mix(vec3(1.0), dotColor, inside);
        fragColor = vec4(outputColor, 1.0);
      }
    `;

    // Clustering fragment shader
    const clusteringFragmentSource = `#version 300 es
      precision highp float;

      in vec2 v_texCoord;
      out vec4 fragColor;

      uniform sampler2D u_video;
      uniform vec2 u_resolution;
      uniform float u_algorithm;
      uniform float u_colorCount;
      uniform float u_threshold;

      vec3 quantize(vec3 color, float levels) {
        float l = max(levels, 2.0);
        return floor(color * l) / (l - 1.0);
      }

      vec3 generateCentroid(int index, int k) {
        float kf = float(k);
        float idx = float(index);
        float divisions = ceil(pow(kf, 1.0/3.0));
        float r_idx = mod(idx, divisions);
        float g_idx = mod(floor(idx / divisions), divisions);
        float b_idx = floor(idx / (divisions * divisions));
        return vec3(
          (r_idx + 0.5) / divisions,
          (g_idx + 0.5) / divisions,
          (b_idx + 0.5) / divisions
        );
      }

      vec3 kmeans(vec3 color, int k) {
        float minDist = 999999.0;
        vec3 bestCentroid = vec3(0.0);
        for (int i = 0; i < 32; i++) {
          if (i >= k) break;
          vec3 centroid = generateCentroid(i, k);
          float dist = length(color - centroid);
          if (dist < minDist) {
            minDist = dist;
            bestCentroid = centroid;
          }
        }
        return bestCentroid;
      }

      vec3 meanshift(vec2 uv, vec3 color, float bandwidth) {
        vec3 currentColor = color;
        vec2 texelSize = 1.0 / u_resolution;
        vec3 weightedSum = vec3(0.0);
        float totalWeight = 0.0;

        for (int dy = -3; dy <= 3; dy++) {
          for (int dx = -3; dx <= 3; dx++) {
            vec2 offset = vec2(float(dx), float(dy)) * texelSize;
            vec3 neighbor = texture(u_video, uv + offset).rgb;
            float colorDist = length(currentColor - neighbor);
            float weight = exp(-(colorDist * colorDist) / (2.0 * bandwidth * bandwidth));
            weightedSum += neighbor * weight;
            totalWeight += weight;
          }
        }

        if (totalWeight > 0.0) {
          currentColor = weightedSum / totalWeight;
        }
        return quantize(currentColor, u_colorCount);
      }

      vec3 posterize(vec2 uv, vec3 color, float levels, float threshold) {
        vec3 quantized = quantize(color, levels);
        vec2 texelSize = 1.0 / u_resolution;
        vec3 weightedSum = vec3(0.0);
        float totalWeight = 0.0;

        for (int dy = -2; dy <= 2; dy++) {
          for (int dx = -2; dx <= 2; dx++) {
            vec2 offset = vec2(float(dx), float(dy)) * texelSize;
            vec3 neighbor = texture(u_video, uv + offset).rgb;
            vec3 neighborQuantized = quantize(neighbor, levels);

            float spatialDist = length(vec2(float(dx), float(dy)));
            float spatialWeight = exp(-spatialDist * spatialDist / 4.0);
            float colorDist = length(quantized - neighborQuantized);
            float colorWeight = 1.0 - smoothstep(0.0, threshold, colorDist);

            float weight = spatialWeight * colorWeight;
            weightedSum += neighborQuantized * weight;
            totalWeight += weight;
          }
        }

        if (totalWeight > 0.0) {
          return weightedSum / totalWeight;
        }
        return quantized;
      }

      void main() {
        vec3 color = texture(u_video, v_texCoord).rgb;
        vec3 result;

        int algo = int(u_algorithm);
        int k = int(u_colorCount);

        if (algo == 0) {
          result = quantize(color, u_colorCount);
        } else if (algo == 1) {
          result = kmeans(color, k);
        } else if (algo == 2) {
          result = meanshift(v_texCoord, color, u_threshold);
        } else {
          result = posterize(v_texCoord, color, u_colorCount, u_threshold);
        }

        fragColor = vec4(result, 1.0);
      }
    `;

    // Passthrough fragment shader
    const passthroughFragmentSource = `#version 300 es
      precision highp float;

      in vec2 v_texCoord;
      out vec4 fragColor;

      uniform sampler2D u_video;

      void main() {
        fragColor = texture(u_video, v_texCoord);
      }
    `;
    END OF OLD INLINE SHADERS */

    // Compile shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const halftoneFragment = this.compileShader(gl.FRAGMENT_SHADER, halftoneFragmentSource);
    const clusteringFragment = this.compileShader(gl.FRAGMENT_SHADER, clusteringFragmentSource);
    const edgesFragment = this.compileShader(gl.FRAGMENT_SHADER, edgesFragmentSource);
    const mosaicFragment = this.compileShader(gl.FRAGMENT_SHADER, mosaicFragmentSource);
    const chromaticFragment = this.compileShader(gl.FRAGMENT_SHADER, chromaticFragmentSource);
    const passthroughFragment = this.compileShader(gl.FRAGMENT_SHADER, passthroughFragmentSource);

    // Create programs
    this.halftoneProgram = this.createProgram(vertexShader, halftoneFragment);
    this.clusteringProgram = this.createProgram(vertexShader, clusteringFragment);
    this.edgesProgram = this.createProgram(vertexShader, edgesFragment);
    this.mosaicProgram = this.createProgram(vertexShader, mosaicFragment);
    this.chromaticProgram = this.createProgram(vertexShader, chromaticFragment);
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

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    const resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const dotSizeLoc = gl.getUniformLocation(program, "u_dotSize");
    const useRandomColorsLoc = gl.getUniformLocation(program, "u_useRandomColors");
    const timeLoc = gl.getUniformLocation(program, "u_time");

    const time = Math.floor(performance.now() / 1000);

    gl.uniform1i(videoLoc, 0);
    gl.uniform2f(resolutionLoc, video.videoWidth, video.videoHeight);
    gl.uniform1f(dotSizeLoc, dotSize);
    gl.uniform1f(useRandomColorsLoc, useRandomColors ? 1.0 : 0.0);
    gl.uniform1f(timeLoc, time);

    // Draw
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(1, 1, 1, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  renderPassthrough(video) {
    const gl = this.gl;
    const program = this.passthroughProgram;

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    gl.uniform1i(videoLoc, 0);

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

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    const resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const algorithmLoc = gl.getUniformLocation(program, "u_algorithm");
    const colorCountLoc = gl.getUniformLocation(program, "u_colorCount");
    const thresholdLoc = gl.getUniformLocation(program, "u_threshold");

    gl.uniform1i(videoLoc, 0);
    gl.uniform2f(resolutionLoc, video.videoWidth, video.videoHeight);
    gl.uniform1f(algorithmLoc, algorithmMap[algorithm] || 0);
    gl.uniform1f(colorCountLoc, colorCount);
    gl.uniform1f(thresholdLoc, threshold);

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

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    const resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const algorithmLoc = gl.getUniformLocation(program, "u_algorithm");
    const thresholdLoc = gl.getUniformLocation(program, "u_threshold");
    const showOverlayLoc = gl.getUniformLocation(program, "u_showOverlay");
    const invertLoc = gl.getUniformLocation(program, "u_invert");
    const edgeColorLoc = gl.getUniformLocation(program, "u_edgeColor");
    const thicknessLoc = gl.getUniformLocation(program, "u_thickness");

    gl.uniform1i(videoLoc, 0);
    gl.uniform2f(resolutionLoc, video.videoWidth, video.videoHeight);
    gl.uniform1f(algorithmLoc, algorithmMap[algorithm] || 0);
    gl.uniform1f(thresholdLoc, threshold);
    gl.uniform1f(showOverlayLoc, showOverlay ? 1.0 : 0.0);
    gl.uniform1f(invertLoc, invert ? 1.0 : 0.0);
    gl.uniform3f(edgeColorLoc, edgeColor[0], edgeColor[1], edgeColor[2]);
    gl.uniform1f(thicknessLoc, thickness);

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

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    const resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const blockSizeLoc = gl.getUniformLocation(program, "u_blockSize");
    const modeLoc = gl.getUniformLocation(program, "u_mode");
    const timeLoc = gl.getUniformLocation(program, "u_time");

    const time = Math.floor(performance.now() / 1000);

    gl.uniform1i(videoLoc, 0);
    gl.uniform2f(resolutionLoc, video.videoWidth, video.videoHeight);
    gl.uniform1f(blockSizeLoc, blockSize);
    gl.uniform1f(modeLoc, modeMap[mode] || 0);
    gl.uniform1f(timeLoc, time);

    // Draw
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

    // Update video texture
    this.updateVideoTexture(video);

    // Use program
    gl.useProgram(program);

    // Set up attributes
    const positionLoc = gl.getAttribLocation(program, "a_position");
    const texCoordLoc = gl.getAttribLocation(program, "a_texCoord");

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLoc);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    const videoLoc = gl.getUniformLocation(program, "u_video");
    const resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    const intensityLoc = gl.getUniformLocation(program, "u_intensity");
    const modeLoc = gl.getUniformLocation(program, "u_mode");
    const centerXLoc = gl.getUniformLocation(program, "u_centerX");
    const centerYLoc = gl.getUniformLocation(program, "u_centerY");

    gl.uniform1i(videoLoc, 0);
    gl.uniform2f(resolutionLoc, video.videoWidth, video.videoHeight);
    gl.uniform1f(intensityLoc, intensity);
    gl.uniform1f(modeLoc, modeMap[mode] || 0);
    gl.uniform1f(centerXLoc, centerX);
    gl.uniform1f(centerYLoc, centerY);

    // Draw
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
      if (this.passthroughProgram) gl.deleteProgram(this.passthroughProgram);
    }
  }
}

export async function initWebGL(canvas) {
  const renderer = new WebGLRenderer();
  await renderer.init(canvas);
  return renderer;
}
