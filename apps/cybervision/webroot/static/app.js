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
    this.trueColorsCheckbox = document.getElementById("trueColorsCheckbox");
    this.colorCountSlider = document.getElementById("colorCountSlider");
    this.colorCountValue = document.getElementById("colorCountValue");
    this.thresholdSlider = document.getElementById("thresholdSlider");
    this.thresholdValue = document.getElementById("thresholdValue");

    // Edge detection controls
    this.edgesControls = document.getElementById("edgesControls");
    this.edgeAlgorithm = document.getElementById("edgeAlgorithm");
    this.edgeThreshold = document.getElementById("edgeThreshold");
    this.edgeThresholdValue = document.getElementById("edgeThresholdValue");
    this.edgeOverlay = document.getElementById("edgeOverlay");
    this.edgeInvert = document.getElementById("edgeInvert");
    this.edgeColor = document.getElementById("edgeColor");
    this.edgeThickness = document.getElementById("edgeThickness");
    this.edgeThicknessValue = document.getElementById("edgeThicknessValue");

    // Mosaic controls
    this.mosaicControls = document.getElementById("mosaicControls");
    this.mosaicBlockSize = document.getElementById("mosaicBlockSize");
    this.mosaicBlockSizeValue = document.getElementById("mosaicBlockSizeValue");
    this.mosaicMode = document.getElementById("mosaicMode");
    this.mosaicInfo = document.getElementById("mosaicInfo");

    // Chromatic aberration controls
    this.chromaticControls = document.getElementById("chromaticControls");
    this.chromaticIntensity = document.getElementById("chromaticIntensity");
    this.chromaticIntensityValue = document.getElementById("chromaticIntensityValue");
    this.chromaticMode = document.getElementById("chromaticMode");
    this.chromaticCenterX = document.getElementById("chromaticCenterX");
    this.chromaticCenterXValue = document.getElementById("chromaticCenterXValue");
    this.chromaticCenterY = document.getElementById("chromaticCenterY");
    this.chromaticCenterYValue = document.getElementById("chromaticCenterYValue");

    // Glitch controls
    this.glitchControls = document.getElementById("glitchControls");
    this.glitchMode = document.getElementById("glitchMode");
    this.glitchIntensity = document.getElementById("glitchIntensity");
    this.glitchIntensityValue = document.getElementById("glitchIntensityValue");
    this.glitchBlockSize = document.getElementById("glitchBlockSize");
    this.glitchBlockSizeValue = document.getElementById("glitchBlockSizeValue");
    this.glitchColorShift = document.getElementById("glitchColorShift");
    this.glitchColorShiftValue = document.getElementById("glitchColorShiftValue");
    this.glitchNoise = document.getElementById("glitchNoise");
    this.glitchNoiseValue = document.getElementById("glitchNoiseValue");
    this.glitchScanline = document.getElementById("glitchScanline");
    this.glitchScanlineValue = document.getElementById("glitchScanlineValue");

    // Thermal controls
    this.thermalControls = document.getElementById("thermalControls");
    this.thermalPalette = document.getElementById("thermalPalette");
    this.thermalContrast = document.getElementById("thermalContrast");
    this.thermalContrastValue = document.getElementById("thermalContrastValue");
    this.thermalInvert = document.getElementById("thermalInvert");

    // Pixel sort controls
    this.pixelSortControls = document.getElementById("pixelSortControls");
    this.pixelSortAngleMode = document.getElementById("pixelSortAngleMode");
    this.pixelSortDirection = document.getElementById("pixelSortDirection");
    this.pixelSortDirectionGroup = document.getElementById("pixelSortDirectionGroup");
    this.pixelSortAngle = document.getElementById("pixelSortAngle");
    this.pixelSortAngleValue = document.getElementById("pixelSortAngleValue");
    this.pixelSortAngleGroup = document.getElementById("pixelSortAngleGroup");
    this.pixelSortThresholdMode = document.getElementById("pixelSortThresholdMode");
    this.pixelSortThresholdLow = document.getElementById("pixelSortThresholdLow");
    this.pixelSortThresholdLowValue = document.getElementById("pixelSortThresholdLowValue");
    this.pixelSortThresholdHigh = document.getElementById("pixelSortThresholdHigh");
    this.pixelSortThresholdHighValue = document.getElementById("pixelSortThresholdHighValue");
    this.pixelSortKey = document.getElementById("pixelSortKey");
    this.pixelSortOrder = document.getElementById("pixelSortOrder");
    this.pixelSortAlgorithm = document.getElementById("pixelSortAlgorithm");
    this.pixelSortIterations = document.getElementById("pixelSortIterations");
    this.pixelSortIterationsValue = document.getElementById("pixelSortIterationsValue");
    this.pixelSortIterationsGroup = document.getElementById("pixelSortIterationsGroup");

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
    this.clusteringAlgorithm = "quantization-kmeans";
    this.useTrueColors = false;
    this.colorCount = 8;
    this.colorThreshold = 0.1;

    // Edge detection state
    this.edgeAlgorithmValue = "sobel";
    this.edgeThresholdValue_state = 0.1;
    this.edgeOverlayValue = false;
    this.edgeInvertValue = false;
    this.edgeColorValue = "#ffffff";
    this.edgeThicknessValue_state = 1;

    // Mosaic state
    this.mosaicBlockSizeValue_state = 8;
    this.mosaicModeValue = "center";

    // Chromatic aberration state
    this.chromaticIntensityValue_state = 10;
    this.chromaticModeValue = "radial";
    this.chromaticCenterXValue_state = 50;
    this.chromaticCenterYValue_state = 50;

    // Glitch state
    this.glitchModeValue = "slices";
    this.glitchIntensityValue_state = 12;
    this.glitchBlockSizeValue_state = 24;
    this.glitchColorShiftValue_state = 4;
    this.glitchNoiseValue_state = 0.15;
    this.glitchScanlineValue_state = 0.3;

    // Thermal state
    this.thermalPaletteValue = "classic";
    this.thermalContrastValue_state = 1.0;
    this.thermalInvertValue = false;

    // Pixel sort state
    this.pixelSortAngleModeValue = "preset";
    this.pixelSortDirectionValue = "horizontal";
    this.pixelSortAngleValue_state = 0;
    this.pixelSortThresholdModeValue = "brightness";
    this.pixelSortThresholdLowValue_state = 0.25;
    this.pixelSortThresholdHighValue_state = 0.75;
    this.pixelSortKeyValue = "luminance";
    this.pixelSortOrderValue = "ascending";
    this.pixelSortAlgorithmValue = "bitonic";
    this.pixelSortIterationsValue_state = 50;

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

    this.trueColorsCheckbox.addEventListener("change", (e) => {
      this.useTrueColors = e.target.checked;
    });

    this.colorCountSlider.addEventListener("input", (e) => {
      this.colorCount = parseInt(e.target.value, 10);
      this.colorCountValue.textContent = this.colorCount;
    });

    this.thresholdSlider.addEventListener("input", (e) => {
      this.colorThreshold = parseFloat(e.target.value);
      this.thresholdValue.textContent = this.colorThreshold.toFixed(2);
    });

    // Edge detection event listeners
    this.edgeAlgorithm.addEventListener("change", (e) => {
      this.edgeAlgorithmValue = e.target.value;
    });

    this.edgeThreshold.addEventListener("input", (e) => {
      this.edgeThresholdValue_state = parseFloat(e.target.value);
      this.edgeThresholdValue.textContent = this.edgeThresholdValue_state.toFixed(2);
    });

    this.edgeOverlay.addEventListener("change", (e) => {
      this.edgeOverlayValue = e.target.checked;
    });

    this.edgeInvert.addEventListener("change", (e) => {
      this.edgeInvertValue = e.target.checked;
    });

    this.edgeColor.addEventListener("input", (e) => {
      this.edgeColorValue = e.target.value;
    });

    this.edgeThickness.addEventListener("input", (e) => {
      this.edgeThicknessValue_state = parseInt(e.target.value, 10);
      this.edgeThicknessValue.textContent = this.edgeThicknessValue_state;
    });

    // Mosaic event listeners
    this.mosaicBlockSize.addEventListener("input", (e) => {
      this.mosaicBlockSizeValue_state = parseInt(e.target.value, 10);
      this.mosaicBlockSizeValue.textContent = this.mosaicBlockSizeValue_state;
    });

    this.mosaicMode.addEventListener("change", (e) => {
      this.mosaicModeValue = e.target.value;
      this.updateMosaicInfo();
    });

    // Chromatic aberration event listeners
    this.chromaticIntensity.addEventListener("input", (e) => {
      this.chromaticIntensityValue_state = parseInt(e.target.value, 10);
      this.chromaticIntensityValue.textContent = this.chromaticIntensityValue_state;
    });

    this.chromaticMode.addEventListener("change", (e) => {
      this.chromaticModeValue = e.target.value;
    });

    this.chromaticCenterX.addEventListener("input", (e) => {
      this.chromaticCenterXValue_state = parseInt(e.target.value, 10);
      this.chromaticCenterXValue.textContent = this.chromaticCenterXValue_state;
    });

    this.chromaticCenterY.addEventListener("input", (e) => {
      this.chromaticCenterYValue_state = parseInt(e.target.value, 10);
      this.chromaticCenterYValue.textContent = this.chromaticCenterYValue_state;
    });

    // Glitch event listeners
    this.glitchMode.addEventListener("change", (e) => {
      this.glitchModeValue = e.target.value;
    });

    this.glitchIntensity.addEventListener("input", (e) => {
      this.glitchIntensityValue_state = parseInt(e.target.value, 10);
      this.glitchIntensityValue.textContent = this.glitchIntensityValue_state;
    });

    this.glitchBlockSize.addEventListener("input", (e) => {
      this.glitchBlockSizeValue_state = parseInt(e.target.value, 10);
      this.glitchBlockSizeValue.textContent = this.glitchBlockSizeValue_state;
    });

    this.glitchColorShift.addEventListener("input", (e) => {
      this.glitchColorShiftValue_state = parseInt(e.target.value, 10);
      this.glitchColorShiftValue.textContent = this.glitchColorShiftValue_state;
    });

    this.glitchNoise.addEventListener("input", (e) => {
      this.glitchNoiseValue_state = parseFloat(e.target.value);
      this.glitchNoiseValue.textContent = this.glitchNoiseValue_state.toFixed(2);
    });

    this.glitchScanline.addEventListener("input", (e) => {
      this.glitchScanlineValue_state = parseFloat(e.target.value);
      this.glitchScanlineValue.textContent = this.glitchScanlineValue_state.toFixed(2);
    });

    // Thermal event listeners
    this.thermalPalette.addEventListener("change", (e) => {
      this.thermalPaletteValue = e.target.value;
    });

    this.thermalContrast.addEventListener("input", (e) => {
      this.thermalContrastValue_state = parseFloat(e.target.value);
      this.thermalContrastValue.textContent = this.thermalContrastValue_state.toFixed(1);
    });

    this.thermalInvert.addEventListener("change", (e) => {
      this.thermalInvertValue = e.target.checked;
    });

    // Pixel sort event listeners
    this.pixelSortAngleMode.addEventListener("change", (e) => {
      this.pixelSortAngleModeValue = e.target.value;
      // Show/hide direction vs angle controls
      if (e.target.value === "preset") {
        this.pixelSortDirectionGroup.style.display = "block";
        this.pixelSortAngleGroup.style.display = "none";
      } else {
        this.pixelSortDirectionGroup.style.display = "none";
        this.pixelSortAngleGroup.style.display = "block";
      }
    });

    this.pixelSortDirection.addEventListener("change", (e) => {
      this.pixelSortDirectionValue = e.target.value;
    });

    this.pixelSortAngle.addEventListener("input", (e) => {
      this.pixelSortAngleValue_state = parseInt(e.target.value, 10);
      this.pixelSortAngleValue.textContent = this.pixelSortAngleValue_state;
    });

    this.pixelSortThresholdMode.addEventListener("change", (e) => {
      this.pixelSortThresholdModeValue = e.target.value;
    });

    this.pixelSortThresholdLow.addEventListener("input", (e) => {
      this.pixelSortThresholdLowValue_state = parseFloat(e.target.value);
      this.pixelSortThresholdLowValue.textContent = this.pixelSortThresholdLowValue_state.toFixed(2);
    });

    this.pixelSortThresholdHigh.addEventListener("input", (e) => {
      this.pixelSortThresholdHighValue_state = parseFloat(e.target.value);
      this.pixelSortThresholdHighValue.textContent = this.pixelSortThresholdHighValue_state.toFixed(2);
    });

    this.pixelSortKey.addEventListener("change", (e) => {
      this.pixelSortKeyValue = e.target.value;
    });

    this.pixelSortOrder.addEventListener("change", (e) => {
      this.pixelSortOrderValue = e.target.value;
    });

    this.pixelSortAlgorithm.addEventListener("change", (e) => {
      this.pixelSortAlgorithmValue = e.target.value;
      // Show/hide iterations slider based on algorithm
      if (e.target.value === "bubble") {
        this.pixelSortIterationsGroup.style.display = "block";
      } else {
        this.pixelSortIterationsGroup.style.display = "none";
      }
    });

    this.pixelSortIterations.addEventListener("input", (e) => {
      this.pixelSortIterationsValue_state = parseInt(e.target.value, 10);
      this.pixelSortIterationsValue.textContent = this.pixelSortIterationsValue_state;
    });

    // Renderer toggle
    this.webglToggle.addEventListener("change", async (e) => {
      await this.handleRendererToggle(e.target.checked);
    });
  }

  async handleRendererToggle(forceWebGL) {
    const wasRunning = this.isRunning;

    // Stop camera if running
    if (wasRunning) {
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

      // Update mosaic info visibility based on new renderer
      this.updateMosaicInfo();

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
    this.edgesControls.style.display = this.currentEffect === "edges" ? "block" : "none";
    this.mosaicControls.style.display = this.currentEffect === "mosaic" ? "block" : "none";
    this.chromaticControls.style.display = this.currentEffect === "chromatic" ? "block" : "none";
    this.glitchControls.style.display = this.currentEffect === "glitch" ? "block" : "none";
    this.thermalControls.style.display = this.currentEffect === "thermal" ? "block" : "none";
    this.pixelSortControls.style.display = this.currentEffect === "pixelsort" ? "block" : "none";

    // Update mosaic info when mosaic effect is shown
    if (this.currentEffect === "mosaic") {
      this.updateMosaicInfo();
    }

    // Update pixel sort iterations visibility
    if (this.currentEffect === "pixelsort") {
      if (this.pixelSortAlgorithmValue === "bubble") {
        this.pixelSortIterationsGroup.style.display = "block";
      } else {
        this.pixelSortIterationsGroup.style.display = "none";
      }
    }
  }

  updateMosaicInfo() {
    // Show info text if using dominant mode with WebGL (falls back to centerSample)
    if (this.mosaicModeValue === "dominant" && this.rendererType === "webgl") {
      this.mosaicInfo.style.display = "flex";
    } else {
      this.mosaicInfo.style.display = "none";
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
      } else if (this.currentEffect === "edges") {
        this.renderEdges();
      } else if (this.currentEffect === "mosaic") {
        this.renderMosaic();
      } else if (this.currentEffect === "chromatic") {
        this.renderChromatic();
      } else if (this.currentEffect === "glitch") {
        this.renderGlitch();
      } else if (this.currentEffect === "thermal") {
        this.renderThermal();
      } else if (this.currentEffect === "pixelsort") {
        this.renderPixelSort();
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
    // Compute algorithm string based on base algorithm and true colors toggle
    const algorithmString = this.useTrueColors
      ? `${this.clusteringAlgorithm}-true`
      : this.clusteringAlgorithm;

    this.renderer.renderClustering(
      this.video,
      algorithmString,
      this.colorCount,
      this.colorThreshold
    );
  }

  renderEdges() {
    // Parse edge color from hex to RGB
    const hex = this.edgeColorValue.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;

    this.renderer.renderEdges(
      this.video,
      this.edgeAlgorithmValue,
      this.edgeThresholdValue_state,
      this.edgeOverlayValue,
      this.edgeInvertValue,
      [r, g, b],
      this.edgeThicknessValue_state
    );
  }

  renderMosaic() {
    this.renderer.renderMosaic(
      this.video,
      this.mosaicBlockSizeValue_state,
      this.mosaicModeValue
    );
  }

  renderChromatic() {
    // Convert center percentage to 0-1 range
    const centerX = this.chromaticCenterXValue_state / 100;
    const centerY = this.chromaticCenterYValue_state / 100;

    this.renderer.renderChromatic(
      this.video,
      this.chromaticIntensityValue_state,
      this.chromaticModeValue,
      centerX,
      centerY
    );
  }

  renderGlitch() {
    this.renderer.renderGlitch(
      this.video,
      this.glitchModeValue,
      this.glitchIntensityValue_state,
      this.glitchBlockSizeValue_state,
      this.glitchColorShiftValue_state,
      this.glitchNoiseValue_state,
      this.glitchScanlineValue_state
    );
  }

  renderThermal() {
    this.renderer.renderThermal(
      this.video,
      this.thermalPaletteValue,
      this.thermalContrastValue_state,
      this.thermalInvertValue
    );
  }

  renderPixelSort() {
    // Only WebGPU for now
    if (this.rendererType === "webgpu") {
      this.renderer.renderPixelSort(
        this.video,
        this.pixelSortAngleModeValue,
        this.pixelSortDirectionValue,
        this.pixelSortAngleValue_state,
        this.pixelSortThresholdLowValue_state,
        this.pixelSortThresholdHighValue_state,
        this.pixelSortThresholdModeValue,
        this.pixelSortKeyValue,
        this.pixelSortOrderValue,
        this.pixelSortAlgorithmValue,
        this.pixelSortIterationsValue_state
      );
    }
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
