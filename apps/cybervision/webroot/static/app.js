/* CyberVision - Main application */
// TODO: Add automated tests - E2E tests with Playwright for UI/renderer switching,
// and unit tests for pure logic (FPS calculation, algorithm mapping, etc.)

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
    this.dotSizeSlider = document.getElementById("dotSizeSlider");
    this.dotSizeValue = document.getElementById("dotSizeValue");
    this.randomColorCheckbox = document.getElementById("randomColorCheckbox");
    this.fpsValue = document.getElementById("fpsValue");
    this.latencyValue = document.getElementById("latencyValue");
    this.gpuStatus = document.getElementById("gpuStatus");
    this.webglToggle = document.getElementById("webglToggle");
    this.resolutionValue = document.getElementById("resolutionValue");

    // Clustering controls
    this.halftoneControls = document.getElementById("halftoneControls");
    this.clusteringControls = document.getElementById("clusteringControls");
    this.algorithmSelect = document.getElementById("algorithmSelect");
    this.colorCountSlider = document.getElementById("colorCountSlider");
    this.colorCountValue = document.getElementById("colorCountValue");
    this.thresholdSlider = document.getElementById("thresholdSlider");
    this.thresholdValue = document.getElementById("thresholdValue");

    // State
    this.renderer = null;
    this.rendererType = null; // 'webgpu' or 'webgl'
    this.webgpuAvailable = false;
    this.webglAvailable = false;
    this.stream = null;
    this.isRunning = false;
    this.animationFrame = null;

    // Effect state
    this.currentEffect = "original";
    this.dotSize = 8;
    this.useRandomColors = false;

    // Clustering state
    this.clusteringAlgorithm = "quantization";
    this.colorCount = 8;
    this.colorThreshold = 0.1;

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    // Latency tracking
    this.frameLatencies = [];
    this.lastLatencyUpdate = performance.now();

    this.init();
  }

  async init() {
    // Probe both renderers to check availability
    await this.probeRenderers();

    // Check URL parameters for renderer control
    const urlParams = new URLSearchParams(window.location.search);
    const forceWebGL = urlParams.get('force-webgl') === 'true' || urlParams.get('disable-webgpu') === 'true';

    // Initialize based on URL parameter or toggle state
    const useWebGL = forceWebGL || this.webglToggle.checked;

    if (useWebGL && this.webglAvailable) {
      await this.switchToRenderer('webgl');
    } else if (this.webgpuAvailable) {
      await this.switchToRenderer('webgpu');
    } else if (this.webglAvailable) {
      await this.switchToRenderer('webgl');
    } else {
      this.gpuStatus.textContent = "Not supported";
      this.gpuStatus.style.color = "#f87171";
      this.setStatus("Error: Neither WebGPU nor WebGL2 is available.");
      this.startBtn.disabled = true;
      this.webglToggle.disabled = true;
      return;
    }

    // Update toggle state and availability
    this.webglToggle.checked = this.rendererType === 'webgl';
    this.webglToggle.disabled = !this.webgpuAvailable; // Disable if WebGPU not available

    this.setupEventListeners();
    this.setStatus(`Ready. Using ${this.rendererType.toUpperCase()}. Click 'Start Camera' to begin.`);
  }

  async probeRenderers() {
    // Test WebGPU availability (check if navigator.gpu exists)
    console.log("Probing WebGPU availability...");
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        this.webgpuAvailable = !!adapter;
        console.log(this.webgpuAvailable ? "✓ WebGPU is available" : "✗ WebGPU adapter not available");
      } catch (err) {
        console.log("✗ WebGPU not available:", err.message);
        this.webgpuAvailable = false;
      }
    } else {
      console.log("✗ WebGPU not supported by browser");
      this.webgpuAvailable = false;
    }

    // Test WebGL availability (check if WebGL2 context can be created)
    console.log("Probing WebGL availability...");
    const testCanvas = document.createElement('canvas');
    const gl = testCanvas.getContext('webgl2');
    this.webglAvailable = !!gl;
    console.log(this.webglAvailable ? "✓ WebGL2 is available" : "✗ WebGL2 not available");
  }

  async switchToRenderer(type) {
    console.log(`Switching to ${type.toUpperCase()}...`);

    // Cleanup old renderer if exists
    if (this.renderer && this.renderer.cleanup) {
      this.renderer.cleanup();
    }

    // Need to recreate canvas to switch between WebGPU and WebGL contexts
    // Save canvas dimensions
    const width = this.canvas.width;
    const height = this.canvas.height;

    // Replace canvas
    const newCanvas = document.createElement('canvas');
    newCanvas.id = 'canvas';
    newCanvas.className = this.canvas.className;
    newCanvas.width = width;
    newCanvas.height = height;
    this.canvas.parentNode.replaceChild(newCanvas, this.canvas);
    this.canvas = newCanvas;

    try {
      if (type === 'webgpu') {
        this.renderer = await initGPU(this.canvas);
        this.rendererType = 'webgpu';
        this.gpuStatus.textContent = "WebGPU ✓";
        this.gpuStatus.style.color = "#34d399";
        console.log("✓ WebGPU initialized");
      } else {
        this.renderer = await initWebGL(this.canvas);
        this.rendererType = 'webgl';
        this.gpuStatus.textContent = "WebGL";
        this.gpuStatus.style.color = "#60a5fa";
        console.log("✓ WebGL initialized");
      }
    } catch (err) {
      this.gpuStatus.textContent = "Error";
      this.gpuStatus.style.color = "#f87171";
      throw err;
    }
  }

  setupEventListeners() {
    // Event listeners
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.stopBtn.addEventListener("click", () => this.stopCamera());

    // Radio button event listeners
    this.effectRadios.forEach((radio) => {
      radio.addEventListener("change", (e) => {
        if (e.target.checked) {
          this.currentEffect = e.target.value;
          this.updateEffectControls();
        }
      });
    });

    this.dotSizeSlider.addEventListener("input", (e) => {
      this.dotSize = parseInt(e.target.value, 10);
      this.dotSizeValue.textContent = this.dotSize;
      if (this.rendererType === "webgpu") {
        this.updateHalftoneParams();
      }
    });

    this.randomColorCheckbox.addEventListener("change", (e) => {
      this.useRandomColors = e.target.checked;
    });

    // Clustering event listeners
    this.algorithmSelect.addEventListener("change", (e) => {
      this.clusteringAlgorithm = e.target.value;
    });

    this.colorCountSlider.addEventListener("input", (e) => {
      this.colorCount = parseInt(e.target.value, 10);
      this.colorCountValue.textContent = this.colorCount;
    });

    this.thresholdSlider.addEventListener("input", (e) => {
      this.colorThreshold = parseFloat(e.target.value);
      this.thresholdValue.textContent = this.colorThreshold.toFixed(2);
    });

    // Renderer toggle
    this.webglToggle.addEventListener("change", async (e) => {
      await this.handleRendererToggle(e.target.checked);
    });
  }

  async handleRendererToggle(forceWebGL) {
    const wasRunning = this.isRunning;
    let videoSettings = null;

    // Stop camera if running and save settings
    if (wasRunning) {
      videoSettings = {
        width: this.video.videoWidth,
        height: this.video.videoHeight,
      };
      this.stopCamera();
    }

    // Switch renderer
    const targetRenderer = forceWebGL ? 'webgl' : 'webgpu';

    if (targetRenderer === 'webgpu' && !this.webgpuAvailable) {
      // Can't switch to WebGPU if not available
      this.webglToggle.checked = true;
      return;
    }

    try {
      await this.switchToRenderer(targetRenderer);
      this.setStatus(`Switched to ${this.rendererType.toUpperCase()}. ${wasRunning ? 'Restarting camera...' : 'Ready.'}`);

      // Restart camera if it was running
      if (wasRunning) {
        await this.startCamera();
      }
    } catch (err) {
      this.setStatus(`Error switching renderer: ${err.message}`);
      console.error(err);
    }
  }

  updateEffectControls() {
    // Show/hide effect-specific controls based on current effect
    this.halftoneControls.style.display = this.currentEffect === "halftone" ? "block" : "none";
    this.clusteringControls.style.display = this.currentEffect === "clustering" ? "block" : "none";
  }

  updateHalftoneParams() {
    // Only for WebGPU
    if (this.rendererType === "webgpu") {
      this.renderer.updateDotSize(this.dotSize);
    }
  }

  async startCamera() {
    try {
      this.setStatus("Requesting camera access...");

      // Request camera at 1080p
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
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

  render() {
    if (!this.isRunning) return;

    const frameStart = performance.now();

    try {
      if (this.currentEffect === "halftone") {
        this.renderHalftone();
      } else if (this.currentEffect === "clustering") {
        this.renderClustering();
      } else if (this.currentEffect === "original") {
        this.renderPassthrough();
      }

      const frameEnd = performance.now();
      const latency = frameEnd - frameStart;
      this.frameLatencies.push(latency);

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
      this.renderer.renderHalftone(this.video, this.useRandomColors);
    } else {
      this.renderer.renderHalftone(this.video, this.dotSize, this.useRandomColors);
    }
  }

  renderClustering() {
    this.renderer.renderClustering(
      this.video,
      this.clusteringAlgorithm,
      this.colorCount,
      this.colorThreshold
    );
  }

  renderPassthrough() {
    this.renderer.renderPassthrough(this.video);
  }

  updateFPS() {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastFpsTime;
    const latencyElapsed = now - this.lastLatencyUpdate;

    // Update FPS once per second
    if (elapsed >= 1000) {
      const fps = Math.round((this.frameCount / elapsed) * 1000);
      this.fpsValue.textContent = fps;
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    // Update latency once per second (average of collected samples)
    if (latencyElapsed >= 1000 && this.frameLatencies.length > 0) {
      const avgLatency = this.frameLatencies.reduce((sum, lat) => sum + lat, 0) / this.frameLatencies.length;
      this.latencyValue.textContent = `${avgLatency.toFixed(2)} ms`;
      this.frameLatencies = [];
      this.lastLatencyUpdate = now;
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
