/* WebGL renderer for CyberVision (fallback for browsers without WebGPU) */

export class WebGLRenderer {
  constructor() {
    this.gl = null;
    this.canvas = null;
    this.halftoneProgram = null;
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

  async createShaders() {
    const gl = this.gl;

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
        float r = hash(cellIndex + vec2(seed, 0.0));
        float g = hash(cellIndex + vec2(0.0, seed));
        float b = hash(cellIndex + vec2(seed, seed));
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

    // Compile shaders
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const halftoneFragment = this.compileShader(gl.FRAGMENT_SHADER, halftoneFragmentSource);
    const passthroughFragment = this.compileShader(gl.FRAGMENT_SHADER, passthroughFragmentSource);

    // Create programs
    this.halftoneProgram = this.createProgram(vertexShader, halftoneFragment);
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

  destroy() {
    const gl = this.gl;
    if (gl) {
      if (this.videoTexture) gl.deleteTexture(this.videoTexture);
      if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
      if (this.texCoordBuffer) gl.deleteBuffer(this.texCoordBuffer);
      if (this.halftoneProgram) gl.deleteProgram(this.halftoneProgram);
      if (this.passthroughProgram) gl.deleteProgram(this.passthroughProgram);
    }
  }
}

export async function initWebGL(canvas) {
  const renderer = new WebGLRenderer();
  await renderer.init(canvas);
  return renderer;
}
