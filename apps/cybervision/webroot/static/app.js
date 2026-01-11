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
    this.dotSizeSlider = document.getElementById("dotSizeSlider");
    this.dotSizeValue = document.getElementById("dotSizeValue");
    this.randomColorCheckbox = document.getElementById("randomColorCheckbox");
    this.fpsValue = document.getElementById("fpsValue");
    this.latencyValue = document.getElementById("latencyValue");
    this.gpuStatus = document.getElementById("gpuStatus");
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
        this.setStatus(`Error: WebGPU failed (${err.message}). WebGL2 also failed (${err2.message}).`);
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

    this.setStatus(`Ready. Using ${this.rendererType.toUpperCase()}. Click 'Start Camera' to begin.`);
  }

  updateEffectControls() {
    // Show/hide effect-specific controls based on current effect
    if (this.currentEffect === "halftone") {
      this.halftoneControls.style.display = "block";
      this.clusteringControls.style.display = "none";
    } else if (this.currentEffect === "clustering") {
      this.halftoneControls.style.display = "none";
      this.clusteringControls.style.display = "block";
    } else {
      this.halftoneControls.style.display = "none";
      this.clusteringControls.style.display = "none";
    }
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
      this.latencyValue.textContent = avgLatency.toFixed(2);
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
