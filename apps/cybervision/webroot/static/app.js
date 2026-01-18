/* CyberVision - Main application */
// TODO: Add automated tests - E2E tests with Playwright for UI/renderer switching,
// and unit tests for pure logic (FPS calculation, algorithm mapping, etc.)

import { initGPU } from "./webgpu-renderer.js";
import { initWebGL } from "./webgl-renderer.js";
import { calculateFPS, hexToRGB, average } from "./utils.js";

class CyberVision {
  constructor() {
    // DOM elements
    this.video = document.getElementById("video");
    this.canvas = document.getElementById("canvas");
    this.startBtn = document.getElementById("startBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.statusEl = document.getElementById("status");
    this.effectButtons = document.querySelectorAll('.effect-btn');
    this.tabButtons = document.querySelectorAll('.tab-button');
    this.tabContents = document.querySelectorAll('.tab-content');
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

    // Kaleidoscope controls
    this.kaleidoscopeControls = document.getElementById("kaleidoscopeControls");
    this.segmentsSlider = document.getElementById("segmentsSlider");
    this.segmentsValue = document.getElementById("segmentsValue");
    this.rotationSpeedSlider = document.getElementById("rotationSpeedSlider");
    this.rotationSpeedValue = document.getElementById("rotationSpeedValue");

    // Segmentation controls
    this.segmentationControls = document.getElementById("segmentationControls");
    this.segmentationMode = document.getElementById("segmentationMode");
    this.segmentationBlurRadius = document.getElementById("segmentationBlurRadius");
    this.segmentationBlurRadiusValue = document.getElementById("segmentationBlurRadiusValue");
    this.segmentationBlurGroup = document.getElementById("segmentationBlurGroup");
    this.segmentationBackgroundGroup = document.getElementById("segmentationBackgroundGroup");
    this.segmentationBackgroundUpload = document.getElementById("segmentationBackgroundUpload");
    this.backgroundPreview = document.getElementById("backgroundPreview");
    this.backgroundPreviewImg = document.getElementById("backgroundPreviewImg");
    this.segmentationLoading = document.getElementById("segmentationLoading");
    this.segmentationLoadingText = document.getElementById("segmentationLoadingText");
    this.segmentationSoftEdges = document.getElementById("segmentationSoftEdges");
    this.segmentationGlow = document.getElementById("segmentationGlow");

    // State
    this.renderer = null;
    this.rendererType = null; // 'webgpu' or 'webgl'
    this.webgpuAvailable = false;
    this.webglAvailable = false;
    this.stream = null;
    this.isRunning = false;
    this.animationFrame = null;

    // Effect state
    this.currentEffect = "segmentation";
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

    // Kaleidoscope state
    this.kaleidoscopeSegments = 8;
    this.kaleidoscopeRotationSpeed = 0.0;

    // Segmentation state
    this.segmentationML = null;  // ML inference instance
    this.segmentationModelLoaded = false;
    this.segmentationMode_state = "blackout";
    this.segmentationBlurRadius_state = 10;
    this.segmentationSoftEdges_state = true;
    this.segmentationGlow_state = false;
    this.segmentationMask = null;  // Current mask
    this.segmentationBackgroundImage = null;  // Background image for replace mode
    this.segmentationFrameSkip = 2;  // Run inference every N frames
    this.segmentationFrameCounter = 0;

    // FPS tracking
    this.frameCount = 0;
    this.lastFpsTime = performance.now();

    // Latency tracking
    this.frameLatencies = [];
    this.lastLatencyUpdate = performance.now();

    this.init();
  }

  async init() {
    console.log("=== CyberVision Initialization Starting ===");
    
    // Probe both renderers to check availability
    await this.probeRenderers();

    // Check URL parameters for renderer control
    const urlParams = new URLSearchParams(window.location.search);
    const forceWebGL = urlParams.get('force-webgl') === 'true' || urlParams.get('disable-webgpu') === 'true';
    console.log("Force WebGL:", forceWebGL);
    console.log("WebGL toggle checked:", this.webglToggle.checked);

    // Initialize based on URL parameter or toggle state
    const useWebGL = forceWebGL || this.webglToggle.checked;
    console.log("Will use WebGL:", useWebGL);
    console.log("Available renderers - WebGPU:", this.webgpuAvailable, "WebGL:", this.webglAvailable);

    try {
      if (useWebGL && this.webglAvailable) {
        console.log("=== Path 1: Using WebGL (forced or toggled) ===");
        await this.switchToRenderer('webgl');
      } else if (this.webgpuAvailable) {
        console.log("=== Path 2: Trying WebGPU first ===");
        // Try WebGPU first
        try {
          await this.switchToRenderer('webgpu');
        } catch (webgpuErr) {
          console.warn("WebGPU initialization failed, falling back to WebGL:", webgpuErr);
          // Fall back to WebGL if WebGPU fails
          if (this.webglAvailable) {
            console.log("Falling back to WebGL...");
            await this.switchToRenderer('webgl');
            this.webgpuAvailable = false; // Mark WebGPU as unavailable
          } else {
            console.error("WebGL not available, cannot fall back");
            throw webgpuErr; // Re-throw if WebGL not available
          }
        }
      } else if (this.webglAvailable) {
        console.log("=== Path 3: Using WebGL (WebGPU not available) ===");
        await this.switchToRenderer('webgl');
      } else {
        console.error("=== Path 4: No renderers available ===");
        this.gpuStatus.textContent = "Not supported";
        this.gpuStatus.style.color = "#f87171";
        this.setStatus("Error: Neither WebGPU nor WebGL2 is available.");
        this.startBtn.disabled = true;
        this.webglToggle.disabled = true;
        return;
      }
    } catch (err) {
      console.error("=== Renderer initialization failed ===", err);
      this.gpuStatus.textContent = "Error";
      this.gpuStatus.style.color = "#f87171";
      this.setStatus(`Error: Failed to initialize renderer: ${err.message}`);
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
    console.log("=== Probing WebGPU availability ===");
    console.log("navigator.gpu exists:", !!navigator.gpu);
    if (navigator.gpu) {
      try {
        console.log("Requesting WebGPU adapter...");
        const adapter = await navigator.gpu.requestAdapter();
        console.log("Adapter received:", adapter);
        this.webgpuAvailable = !!adapter;
        console.log(this.webgpuAvailable ? "✓ WebGPU is available" : "✗ WebGPU adapter not available");
      } catch (err) {
        console.log("✗ WebGPU not available:", err);
        this.webgpuAvailable = false;
      }
    } else {
      console.log("✗ WebGPU not supported by browser");
      this.webgpuAvailable = false;
    }

    // Test WebGL availability (check if WebGL2 context can be created)
    console.log("=== Probing WebGL availability ===");
    const testCanvas = document.createElement('canvas');
    console.log("Test canvas created:", testCanvas);
    const gl = testCanvas.getContext('webgl2');
    console.log("WebGL2 context:", gl);
    this.webglAvailable = !!gl;
    console.log(this.webglAvailable ? "✓ WebGL2 is available" : "✗ WebGL2 not available");
    console.log("=== Probe complete: WebGPU=" + this.webgpuAvailable + ", WebGL=" + this.webglAvailable + " ===");
  }

  async switchToRenderer(type) {
    console.log(`=== Switching to ${type.toUpperCase()} ===`);

    // Cleanup old renderer if exists
    if (this.renderer && this.renderer.cleanup) {
      console.log("Cleaning up old renderer");
      this.renderer.cleanup();
    }

    // Need to recreate canvas to switch between WebGPU and WebGL contexts
    // Save canvas dimensions
    const width = this.canvas.width;
    const height = this.canvas.height;
    console.log("Canvas dimensions:", width, "x", height);

    // Replace canvas
    const newCanvas = document.createElement('canvas');
    newCanvas.id = 'canvas';
    newCanvas.className = this.canvas.className;
    newCanvas.width = width;
    newCanvas.height = height;
    this.canvas.parentNode.replaceChild(newCanvas, this.canvas);
    this.canvas = newCanvas;
    console.log("Canvas replaced");

    try {
      if (type === 'webgpu') {
        console.log("Calling initGPU...");
        this.renderer = await initGPU(this.canvas);
        this.rendererType = 'webgpu';
        this.gpuStatus.textContent = "WebGPU ✓";
        this.gpuStatus.style.color = "#34d399";
        console.log("✓ WebGPU initialized successfully");
      } else {
        console.log("Calling initWebGL...");
        this.renderer = await initWebGL(this.canvas);
        this.rendererType = 'webgl';
        this.gpuStatus.textContent = "WebGL";
        this.gpuStatus.style.color = "#60a5fa";
        console.log("✓ WebGL initialized successfully");
      }
      console.log("Renderer object:", this.renderer);
    } catch (err) {
      console.error("Renderer initialization error:", err);
      this.gpuStatus.textContent = "Error";
      this.gpuStatus.style.color = "#f87171";
      throw err;
    }
  }

  setupEventListeners() {
    // Event listeners
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.stopBtn.addEventListener("click", () => this.stopCamera());

    // Tab switching event listeners
    this.tabButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        const targetTab = e.currentTarget.dataset.tab;

        // Remove active class from all tabs and content
        this.tabButtons.forEach((btn) => btn.classList.remove("active"));
        this.tabContents.forEach((content) => content.classList.remove("active"));

        // Add active class to clicked tab and corresponding content
        e.currentTarget.classList.add("active");
        document.getElementById(`tab-${targetTab}`).classList.add("active");
      });
    });

    // Effect button event listeners
    this.effectButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        // Remove selected from all effect buttons across all tabs
        this.effectButtons.forEach(b => {
          b.classList.remove('selected');
          b.setAttribute('aria-checked', 'false');
        });
        // Add selected to clicked button
        btn.classList.add('selected');
        btn.setAttribute('aria-checked', 'true');
        this.currentEffect = btn.dataset.effect;
        this.updateEffectControls();
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
      this.updatePixelSortIterationsVisibility();
    });

    this.pixelSortIterations.addEventListener("input", (e) => {
      this.pixelSortIterationsValue_state = parseInt(e.target.value, 10);
      this.pixelSortIterationsValue.textContent = this.pixelSortIterationsValue_state;
    });

    // Kaleidoscope event listeners
    this.segmentsSlider.addEventListener("input", (e) => {
      this.kaleidoscopeSegments = parseInt(e.target.value, 10);
      this.segmentsValue.textContent = this.kaleidoscopeSegments;
    });

    this.rotationSpeedSlider.addEventListener("input", (e) => {
      this.kaleidoscopeRotationSpeed = parseFloat(e.target.value);
      this.rotationSpeedValue.textContent = this.kaleidoscopeRotationSpeed.toFixed(2);
    });

    // Segmentation event listeners
    this.segmentationMode.addEventListener("change", (e) => {
      this.segmentationMode_state = e.target.value;
      this.updateSegmentationControlsVisibility();
    });

    this.segmentationBlurRadius.addEventListener("input", (e) => {
      this.segmentationBlurRadius_state = parseInt(e.target.value, 10);
      this.segmentationBlurRadiusValue.textContent = this.segmentationBlurRadius_state;
    });

    this.segmentationBackgroundUpload.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          const img = new Image();
          img.onload = () => {
            this.segmentationBackgroundImage = img;
            this.backgroundPreviewImg.src = event.target.result;
            this.backgroundPreview.style.display = "block";
            // Update background texture in renderer
            if (this.renderer && this.renderer.updateBackgroundImage) {
              this.renderer.updateBackgroundImage(img);
            }
          };
          img.src = event.target.result;
        };
        reader.readAsDataURL(file);
      }
    });

    // Segmentation soft edges and glow checkboxes
    this.segmentationSoftEdges.addEventListener("change", (e) => {
      this.segmentationSoftEdges_state = e.target.checked;
    });

    this.segmentationGlow.addEventListener("change", (e) => {
      this.segmentationGlow_state = e.target.checked;
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
    this.kaleidoscopeControls.style.display = this.currentEffect === "kaleidoscope" ? "block" : "none";
    this.segmentationControls.style.display = this.currentEffect === "segmentation" ? "block" : "none";

    // Update mosaic info when mosaic effect is shown
    if (this.currentEffect === "mosaic") {
      this.updateMosaicInfo();
    }

    // Update pixel sort iterations visibility
    if (this.currentEffect === "pixelsort") {
      this.updatePixelSortIterationsVisibility();
    }

    // Load ML model when segmentation effect is shown
    if (this.currentEffect === "segmentation") {
      this.loadSegmentationModel();
      this.updateSegmentationControlsVisibility();
    }
  }

  updatePixelSortIterationsVisibility() {
    // Show iterations slider only for bubble sort algorithm
    if (this.pixelSortAlgorithmValue === "bubble") {
      this.pixelSortIterationsGroup.style.display = "block";
    } else {
      this.pixelSortIterationsGroup.style.display = "none";
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

  updateSegmentationControlsVisibility() {
    // Show/hide blur or background controls based on mode
    if (this.segmentationMode_state === "blur") {
      this.segmentationBlurGroup.style.display = "block";
      this.segmentationBackgroundGroup.style.display = "none";
    } else if (this.segmentationMode_state === "replace") {
      this.segmentationBlurGroup.style.display = "none";
      this.segmentationBackgroundGroup.style.display = "block";
    } else {
      // blackout mode
      this.segmentationBlurGroup.style.display = "none";
      this.segmentationBackgroundGroup.style.display = "none";
    }
  }

  async loadSegmentationModel() {
    // Only load once
    if (this.segmentationModelLoaded || this.segmentationML) {
      return;
    }

    try {
      this.segmentationLoadingText.textContent = "Loading ML model...";
      this.segmentationLoading.style.display = "flex";

      // Dynamically import ML inference class
      const { PortraitSegmentation } = await import("./ml-inference.js");
      this.segmentationML = new PortraitSegmentation();

      // Load model with progress callback
      await this.segmentationML.loadModel(
        "/static/models/segmentation.onnx",
        (progress) => {
          if (progress.stage === "downloading") {
            const percent = Math.round(progress.progress * 100);
            this.segmentationLoadingText.textContent = `Downloading model... ${percent}%`;
          } else if (progress.stage === "initializing") {
            this.segmentationLoadingText.textContent = "Initializing model...";
          } else if (progress.stage === "ready") {
            this.segmentationLoadingText.textContent = "Model ready!";
            this.segmentationLoading.style.display = "none";
          }
        }
      );

      this.segmentationModelLoaded = true;
      console.log("Segmentation model loaded successfully");
    } catch (error) {
      console.error("Failed to load segmentation model:", error);
      this.segmentationLoadingText.textContent = `Error loading model: ${error.message}`;
      // Keep loading indicator visible to show error
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
      } else if (this.currentEffect === "kaleidoscope") {
        this.renderKaleidoscope();
      } else if (this.currentEffect === "segmentation") {
        this.renderSegmentation();
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
    const rgb = hexToRGB(this.edgeColorValue);

    this.renderer.renderEdges(
      this.video,
      this.edgeAlgorithmValue,
      this.edgeThresholdValue_state,
      this.edgeOverlayValue,
      this.edgeInvertValue,
      rgb,
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
      const fps = calculateFPS(this.frameCount, elapsed);
      this.fpsValue.textContent = fps;
      this.frameCount = 0;
      this.lastFpsTime = now;
    }

    // Update latency once per second (average of collected samples)
    if (latencyElapsed >= 1000 && this.frameLatencies.length > 0) {
      const avgLatency = average(this.frameLatencies);
      this.latencyValue.textContent = `${avgLatency.toFixed(2)} ms`;
      this.frameLatencies = [];
      this.lastLatencyUpdate = now;
    }
  }

  renderKaleidoscope() {
    this.renderer.renderKaleidoscope(
      this.video,
      this.kaleidoscopeSegments,
      this.kaleidoscopeRotationSpeed
    );
  }

  async renderSegmentation() {
    // Only render with WebGPU (segmentation not supported in WebGL yet)
    if (this.rendererType !== "webgpu") {
      // Fall back to passthrough
      this.renderPassthrough();
      return;
    }

    // Check if model is loaded
    if (!this.segmentationModelLoaded || !this.segmentationML) {
      // Show passthrough while loading
      this.renderPassthrough();
      return;
    }

    // Run inference with frame skipping for performance
    this.segmentationFrameCounter++;
    if (this.segmentationFrameCounter >= this.segmentationFrameSkip || !this.segmentationMask) {
      try {
        // Run segmentation inference
        const maskData = await this.segmentationML.segmentFrame(this.video);

        // Postprocess mask
        this.segmentationMask = this.segmentationML.postprocessMask(maskData);

        this.segmentationFrameCounter = 0;
      } catch (error) {
        console.error("Segmentation inference error:", error);
        // Continue with old mask or passthrough
      }
    }

    // Render with current mask
    if (this.segmentationMask) {
      this.renderer.renderSegmentation(
        this.video,
        this.segmentationMode_state,
        this.segmentationBlurRadius_state,
        this.segmentationMask,
        this.segmentationSoftEdges_state,
        this.segmentationGlow_state
      );
    } else {
      // No mask yet, show passthrough
      this.renderPassthrough();
    }
  }

  setStatus(text) {
    this.statusEl.textContent = text;
  }
}

// Initialize app when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    window.cyberVisionApp = new CyberVision();
  });
} else {
  window.cyberVisionApp = new CyberVision();
}
