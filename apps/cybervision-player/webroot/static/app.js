/* CyberVision Video Player - Main application */

// Import renderers from shared library
import { WebGPURenderer } from "/lib/webgpu-renderer.js";
import { WebGLRenderer } from "/lib/webgl-renderer.js";

class VideoPlayer {
  constructor() {
    // DOM elements
    this.videoPathInput = document.getElementById("video-path");
    this.loadVideoBtn = document.getElementById("load-video-btn");
    this.videoElement = document.getElementById("video-element");
    this.canvas = document.getElementById("video-canvas");
    this.playPauseBtn = document.getElementById("play-pause-btn");
    this.seekSlider = document.getElementById("seek-slider");
    this.timeDisplay = document.getElementById("time-display");
    this.effectSelect = document.getElementById("effect-select");
    this.effectParams = document.getElementById("effect-params");
    this.statusMessage = document.getElementById("status-message");

    // State
    this.renderer = null;
    this.rendererType = null;
    this.isPlaying = false;
    this.isVideoLoaded = false;
    this.animationFrame = null;
    this.currentEffect = "original";

    // Effect parameters
    this.dotSize = 8;
    this.useRandomColors = false;
    this.clusteringAlgorithm = "quantization-kmeans";
    this.colorCount = 8;
    this.segments = 6;
    this.rotationSpeed = 1.0;

    this.init();
  }

  init() {
    // Event listeners
    this.loadVideoBtn.addEventListener("click", () => this.loadVideo());
    this.playPauseBtn.addEventListener("click", () => this.togglePlayPause());
    this.seekSlider.addEventListener("input", () => this.seek());
    this.effectSelect.addEventListener("change", () => this.changeEffect());

    // Video event listeners
    this.videoElement.addEventListener("loadedmetadata", () => this.onVideoLoaded());
    this.videoElement.addEventListener("timeupdate", () => this.updateTime());
    this.videoElement.addEventListener("ended", () => this.onVideoEnded());
    this.videoElement.addEventListener("error", (e) => this.onVideoError(e));

    this.showStatus("Ready. Enter a video path to begin.");
  }

  async loadVideo() {
    const videoPath = this.videoPathInput.value.trim();
    if (!videoPath) {
      this.showStatus("Please enter a video file path", "error");
      return;
    }

    this.showStatus("Loading video...");

    try {
      // Set video source
      const encodedPath = encodeURIComponent(videoPath);
      this.videoElement.src = `/api/video?path=${encodedPath}`;

      // Wait for video to load
      await new Promise((resolve, reject) => {
        const loadHandler = () => {
          this.videoElement.removeEventListener("loadedmetadata", loadHandler);
          this.videoElement.removeEventListener("error", errorHandler);
          resolve();
        };
        const errorHandler = (e) => {
          this.videoElement.removeEventListener("loadedmetadata", loadHandler);
          this.videoElement.removeEventListener("error", errorHandler);
          reject(e);
        };
        this.videoElement.addEventListener("loadedmetadata", loadHandler);
        this.videoElement.addEventListener("error", errorHandler);
      });

    } catch (error) {
      this.showStatus(`Failed to load video: ${error.message}`, "error");
      console.error("Video load error:", error);
    }
  }

  async onVideoLoaded() {
    console.log("Video loaded:", {
      width: this.videoElement.videoWidth,
      height: this.videoElement.videoHeight,
      duration: this.videoElement.duration
    });

    // Set canvas size to match video
    this.canvas.width = this.videoElement.videoWidth;
    this.canvas.height = this.videoElement.videoHeight;

    // Initialize renderer
    try {
      await this.initRenderer();
      this.isVideoLoaded = true;
      this.playPauseBtn.disabled = false;
      this.seekSlider.disabled = false;
      this.showStatus("Video loaded. Click Play to start.");
    } catch (error) {
      this.showStatus(`Failed to initialize renderer: ${error.message}`, "error");
      console.error("Renderer init error:", error);
    }
  }

  async initRenderer() {
    // Try WebGPU first, fallback to WebGL
    try {
      this.renderer = new WebGPURenderer();
      await this.renderer.init(this.canvas);
      await this.renderer.setupPipeline(this.videoElement, this.dotSize);
      this.rendererType = "webgpu";
      console.log("Using WebGPU renderer");
    } catch (error) {
      console.warn("WebGPU not available, falling back to WebGL:", error);
      try {
        this.renderer = new WebGLRenderer();
        await this.renderer.init(this.canvas);
        await this.renderer.setupPipeline(this.videoElement);
        this.rendererType = "webgl";
        console.log("Using WebGL renderer");
      } catch (webglError) {
        throw new Error("Neither WebGPU nor WebGL available");
      }
    }
  }

  togglePlayPause() {
    if (!this.isVideoLoaded) return;

    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    this.videoElement.play();
    this.isPlaying = true;
    this.playPauseBtn.textContent = "Pause";
    this.startRenderLoop();
    this.showStatus("Playing...");
  }

  pause() {
    this.videoElement.pause();
    this.isPlaying = false;
    this.playPauseBtn.textContent = "Play";
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.showStatus("Paused");
  }

  seek() {
    if (!this.isVideoLoaded) return;
    const seekTime = (this.seekSlider.value / 100) * this.videoElement.duration;
    this.videoElement.currentTime = seekTime;
  }

  updateTime() {
    if (!this.isVideoLoaded) return;

    const currentTime = this.formatTime(this.videoElement.currentTime);
    const duration = this.formatTime(this.videoElement.duration);
    this.timeDisplay.textContent = `${currentTime} / ${duration}`;

    // Update seek slider
    const progress = (this.videoElement.currentTime / this.videoElement.duration) * 100;
    this.seekSlider.value = progress || 0;
  }

  formatTime(seconds) {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  onVideoEnded() {
    this.isPlaying = false;
    this.playPauseBtn.textContent = "Play";
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    this.showStatus("Playback ended");
  }

  onVideoError(e) {
    console.error("Video error:", e);
    const error = this.videoElement.error;
    let message = "Unknown error";
    if (error) {
      switch (error.code) {
        case error.MEDIA_ERR_ABORTED:
          message = "Video loading aborted";
          break;
        case error.MEDIA_ERR_NETWORK:
          message = "Network error";
          break;
        case error.MEDIA_ERR_DECODE:
          message = "Video decode error";
          break;
        case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
          message = "Video format not supported";
          break;
      }
    }
    this.showStatus(`Video error: ${message}`, "error");
  }

  startRenderLoop() {
    const render = () => {
      if (!this.isPlaying) return;

      try {
        this.renderFrame();
      } catch (error) {
        console.error("Render error:", error);
        this.showStatus(`Render error: ${error.message}`, "error");
        this.pause();
        return;
      }

      this.animationFrame = requestAnimationFrame(render);
    };

    render();
  }

  renderFrame() {
    if (!this.renderer || !this.videoElement.videoWidth) return;

    switch (this.currentEffect) {
      case "original":
        this.renderer.renderPassthrough(this.videoElement);
        break;
      case "halftone":
        this.renderer.renderHalftone(this.videoElement, this.useRandomColors);
        break;
      case "clustering":
        this.renderer.renderClustering(this.videoElement, this.clusteringAlgorithm, this.colorCount, 0.1);
        break;
      case "edges":
        this.renderer.renderEdges(this.videoElement, "sobel", 0.3, false, false, [0, 0, 0], 1);
        break;
      case "mosaic":
        this.renderer.renderMosaic(this.videoElement, 16, "average");
        break;
      case "chromatic":
        this.renderer.renderChromatic(this.videoElement, "radial", 0.02, 0.5, 0.5);
        break;
      case "glitch":
        this.renderer.renderGlitch(this.videoElement, "rgb-shift", 0.5, 8, 0.3, 0.2, 0.3);
        break;
      case "thermal":
        this.renderer.renderThermal(this.videoElement, "iron", 1.0, false);
        break;
      case "pixelsort":
        this.renderer.renderPixelSort(this.videoElement, "horizontal", 0, "brightness", 0.2, 0.8, "hue", "ascending", "bubble", 1);
        break;
      case "kaleidoscope":
        this.renderer.renderKaleidoscope(this.videoElement, this.segments, this.rotationSpeed);
        break;
      case "segmentation":
        // Segmentation requires ML model, skip for now
        this.renderer.renderPassthrough(this.videoElement);
        break;
      default:
        this.renderer.renderPassthrough(this.videoElement);
    }
  }

  changeEffect() {
    this.currentEffect = this.effectSelect.value;
    this.showStatus(`Effect: ${this.currentEffect}`);
    // Update parameter controls based on effect
    this.updateEffectParams();
  }

  updateEffectParams() {
    // For now, just clear params. Can add sliders for each effect later
    this.effectParams.innerHTML = "";

    switch (this.currentEffect) {
      case "halftone":
        this.effectParams.innerHTML = `
          <label>
            Dot Size: <input type="range" id="dot-size-slider" min="2" max="20" value="${this.dotSize}">
            <span id="dot-size-value">${this.dotSize}</span>
          </label>
          <label>
            <input type="checkbox" id="random-colors"> Random Colors
          </label>
        `;
        document.getElementById("dot-size-slider")?.addEventListener("input", (e) => {
          this.dotSize = parseInt(e.target.value);
          document.getElementById("dot-size-value").textContent = this.dotSize;
        });
        document.getElementById("random-colors")?.addEventListener("change", (e) => {
          this.useRandomColors = e.target.checked;
        });
        break;

      case "kaleidoscope":
        this.effectParams.innerHTML = `
          <label>
            Segments: <input type="range" id="segments-slider" min="2" max="12" value="${this.segments}">
            <span id="segments-value">${this.segments}</span>
          </label>
          <label>
            Rotation Speed: <input type="range" id="rotation-slider" min="0" max="2" step="0.1" value="${this.rotationSpeed}">
            <span id="rotation-value">${this.rotationSpeed}</span>
          </label>
        `;
        document.getElementById("segments-slider")?.addEventListener("input", (e) => {
          this.segments = parseInt(e.target.value);
          document.getElementById("segments-value").textContent = this.segments;
        });
        document.getElementById("rotation-slider")?.addEventListener("input", (e) => {
          this.rotationSpeed = parseFloat(e.target.value);
          document.getElementById("rotation-value").textContent = this.rotationSpeed;
        });
        break;

      case "clustering":
        this.effectParams.innerHTML = `
          <label>
            Color Count: <input type="range" id="color-count-slider" min="2" max="16" value="${this.colorCount}">
            <span id="color-count-value">${this.colorCount}</span>
          </label>
        `;
        document.getElementById("color-count-slider")?.addEventListener("input", (e) => {
          this.colorCount = parseInt(e.target.value);
          document.getElementById("color-count-value").textContent = this.colorCount;
        });
        break;
    }
  }

  showStatus(message, type = "info") {
    this.statusMessage.textContent = message;
    this.statusMessage.className = type;
  }
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const player = new VideoPlayer();
});
