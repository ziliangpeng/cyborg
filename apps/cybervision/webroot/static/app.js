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
    this.effectRadios = document.querySelectorAll('input[name="effect"]');
    this.resolutionRadios = document.querySelectorAll('input[name="resolution"]');
    this.dotSizeSlider = document.getElementById("dotSizeSlider");
    this.dotSizeValue = document.getElementById("dotSizeValue");
    this.fpsValue = document.getElementById("fpsValue");
    this.gpuStatus = document.getElementById("gpuStatus");
    this.resolutionValue = document.getElementById("resolutionValue");

    // State
    this.renderer = null;
    this.rendererType = null; // 'webgpu' or 'webgl'
    this.stream = null;
    this.isRunning = false;
    this.animationFrame = null;

    // Effect state
    this.currentEffect = "halftone";
    this.dotSize = 8;
    this.selectedResolution = "1080p"; // Default resolution

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    this.init();
  }

  async init() {
    // Check if WebGL fallback is disabled via URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const disableWebGL = urlParams.get('disable-webgl') === 'true';

    // Try WebGPU first, fallback to WebGL (unless disabled)
    try {
      console.log("Attempting to initialize WebGPU...");
      console.log("navigator.gpu:", navigator.gpu);
      this.renderer = await initGPU(this.canvas);
      this.rendererType = "webgpu";
      this.gpuStatus.textContent = "WebGPU ✓";
      this.gpuStatus.style.color = "#34d399";
      console.log("✓ WebGPU initialized successfully");
      console.log("✓ Using compute shaders");
    } catch (err) {
      console.log("✗ WebGPU failed:", err.message);

      if (disableWebGL) {
        this.gpuStatus.textContent = "WebGL disabled";
        this.gpuStatus.style.color = "#f87171";
        this.setStatus(`WebGPU failed and WebGL fallback is disabled. Error: ${err.message}`);
        this.startBtn.disabled = true;
        return;
      }

      console.log("Trying WebGL fallback...");
      try {
        this.renderer = await initWebGL(this.canvas);
        this.rendererType = "webgl";
        this.gpuStatus.textContent = "WebGL (fallback)";
        this.gpuStatus.style.color = "#60a5fa";
        console.log("✓ WebGL initialized successfully");
        console.log("✓ Using fragment shaders");
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

    // Radio button event listeners
    this.effectRadios.forEach((radio) => {
      radio.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.currentEffect = e.target.value;
        }
      });
    });

    this.resolutionRadios.forEach((radio) => {
      radio.addEventListener("change", async (e) => {
        if (e.target.checked) {
          this.selectedResolution = e.target.value;

          // Restart camera if it's already running
          if (this.isRunning) {
            await this.restartCamera();
          }
        }
      });
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
    if (this.rendererType === "webgpu") {
      this.renderer.updateDotSize(this.dotSize);
    }
  }

  getResolutionConstraints() {
    const resolutions = {
      "4k": { width: 3840, height: 2160 },
      "1080p": { width: 1920, height: 1080 },
      "720p": { width: 1280, height: 720 },
    };
    return resolutions[this.selectedResolution];
  }

  async startCamera() {
    try {
      this.setStatus("Requesting camera access...");

      // Get selected resolution
      const resolution = this.getResolutionConstraints();

      // Request camera
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
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

      // Setup canvas size (always match video dimensions, rotation is handled by CSS)
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;

      console.log(`Video dimensions: ${this.video.videoWidth}x${this.video.videoHeight}`);

      // Update resolution display
      this.resolutionValue.textContent = `${this.video.videoWidth}x${this.video.videoHeight}`;

      // Initialize renderer-specific resources
      if (this.rendererType === "webgpu") {
        await this.renderer.setupPipeline(this.video, this.dotSize);
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
    this.resolutionValue.textContent = "-";
    this.setStatus("Camera stopped.");
  }

  async restartCamera() {
    // Store current state
    const wasRunning = this.isRunning;

    // Stop camera
    this.isRunning = false;
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    // Wait a brief moment for cleanup
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Restart if it was running
    if (wasRunning) {
      await this.startCamera();
    }
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
      this.renderer.renderHalftone(this.video);
    } else {
      this.renderer.renderHalftone(this.video, this.dotSize);
    }
  }

  renderPassthrough() {
    this.renderer.renderPassthrough(this.video);
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
