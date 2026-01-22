/* CyberVision - Main application */
// TODO: Add automated tests - E2E tests with Playwright for UI/renderer switching,
// and unit tests for pure logic (FPS calculation, algorithm mapping, etc.)

import { initGPU } from "/lib/webgpu-renderer.js";
import { initWebGL } from "/lib/webgl-renderer.js";
import { calculateFPS, hexToRGB, average } from "/lib/utils.js";
import { Histogram } from "/lib/histogram.js";

class CyberVision {
  constructor() {
    // DOM elements (Camera)
    this.cameraVideoElement = document.getElementById("camera-video-element");
    this.startBtn = document.getElementById("startBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.screenshotBtn = document.getElementById("screenshotBtn");
    this.statusEl = document.getElementById("status"); // Status for camera operations

    // DOM elements (Video Player)
    this.videoFileInput = document.getElementById("videoFile");
    this.chooseFileBtn = document.getElementById("chooseFileBtn");
    this.dropZone = document.getElementById("dropZone");
    this.currentFileDisplay = document.getElementById("currentFile");
    this.videoElement = document.getElementById("video-element"); // Video element for playing files
    this.playPauseBtn = document.getElementById("play-pause-btn");
    this.seekSlider = document.getElementById("seek-slider");
    this.timeDisplay = document.getElementById("time-display");
    this.playerStatusEl = document.getElementById("player-status"); // Status for video player operations

    // Shared DOM elements
    this.canvas = document.getElementById("canvas");
    this.effectButtons = document.querySelectorAll('.effect-btn');
    this.tabButtons = document.querySelectorAll('.effect-tab-button'); // Effect tab buttons
    this.tabContents = document.querySelectorAll('.effect-tab-content'); // Effect tab contents

    // Input source tab elements
    this.inputSourceTabButtons = document.querySelectorAll('.input-source-tab-button');
    this.inputSourceTabContents = document.querySelectorAll('.input-source-tab-content');

    // UI elements
    this.fpsValue = document.getElementById("fpsValue");
    this.latencyValue = document.getElementById("latencyValue");
    this.gpuStatus = document.getElementById("gpuStatus");
    this.webglToggle = document.getElementById("webglToggle");
    this.resolutionValue = document.getElementById("resolutionValue");

    // Halftone controls
    this.halftoneControls = document.getElementById("halftoneControls");
    this.dotSizeSlider = document.getElementById("dotSizeSlider");
    this.dotSizeValue = document.getElementById("dotSizeValue");
    this.randomColorCheckbox = document.getElementById("randomColorCheckbox");

    // Clustering controls
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

    // Histogram
    this.histogramCanvas = document.getElementById("histogramCanvas");
    this.histogram = new Histogram(this.histogramCanvas);

    // State (Shared)
    this.renderer = null;
    this.rendererType = null; // 'webgpu' or 'webgl'
    this.webgpuAvailable = false;
    this.webglAvailable = false;
    this.activeInputSource = 'camera'; // 'camera' or 'video-file'
    this.currentSourceVideo = null; // Reference to the currently active video element (camera or file)

    // State (Camera specific)
    this.stream = null; // Camera stream
    this.isCameraRunning = false;
    this.cameraAnimationFrame = null; // For camera render loop

    // State (Video Player specific)
    this.currentBlobUrl = null;
    this.isVideoPlaying = false;
    this.isVideoLoaded = false;
    this.videoAnimationFrame = null; // For video render loop

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

    // Initialize UI immediately
    this.setupEventListeners();
    this.updateEffectControls();
    this.switchInputSource('camera'); // Initialize with camera as default
    
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
    // Input source tab switching
    this.inputSourceTabButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        const targetTab = e.currentTarget.dataset.tab;
        this.switchInputSource(targetTab);
      });
    });

    // Camera control listeners
    this.startBtn.addEventListener("click", () => this.startCamera());
    this.stopBtn.addEventListener("click", () => this.stopCamera());
    this.screenshotBtn.addEventListener("click", () => this.takeScreenshot());

    // Video Player control listeners
    this.videoFileInput.addEventListener("change", (e) => {
      if (e.target.files[0]) {
        this.loadLocalVideo(e.target.files[0]);
      }
    });

    this.chooseFileBtn.addEventListener("click", () => {
      this.videoFileInput.click();
    });

    this.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropZone.classList.add("drag-over");
    });
    this.dropZone.addEventListener("dragleave", () => {
      this.dropZone.classList.remove("drag-over");
    });
    this.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files[0]) {
        this.loadLocalVideo(e.dataTransfer.files[0]);
      }
    });

    this.playPauseBtn.addEventListener("click", () => this.togglePlayPause());
    this.seekSlider.addEventListener("input", () => this.seek());

    // Video element event listeners
    this.videoElement.addEventListener("loadedmetadata", () => this.onVideoLoaded());
    this.videoElement.addEventListener("timeupdate", () => this.updateTime());
    this.videoElement.addEventListener("ended", () => this.onVideoEnded());
    this.videoElement.addEventListener("error", (e) => this.onVideoError(e));

    // Effect tab switching event listeners
    this.tabButtons.forEach((button) => {
      button.addEventListener("click", (e) => {
        const targetTab = e.currentTarget.dataset.tab;

        this.tabButtons.forEach((btn) => btn.classList.remove("active"));
        this.tabContents.forEach((content) => content.classList.remove("active"));

        e.currentTarget.classList.add("active");
        document.getElementById(`tab-${targetTab}`).classList.add("active");
      });
    });

    // Effect button event listeners
    this.effectButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        this.effectButtons.forEach(b => {
          b.classList.remove('selected');
          b.setAttribute('aria-checked', 'false');
        });
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

  handleRendererToggle(forceWebGL) {
    const wasRunning = this.isCameraRunning || this.isVideoPlaying;

    // Stop current source if running
    if (this.isCameraRunning) {
      this.stopCamera();
    } else if (this.isVideoPlaying) {
      this.pauseVideo();
    }

    // Switch renderer
    const targetRenderer = forceWebGL ? 'webgl' : 'webgpu';

    if (targetRenderer === 'webgpu' && !this.webgpuAvailable) {
      // Can't switch to WebGPU if not available
      this.webglToggle.checked = true;
      return;
    }

    try {
      this.switchToRenderer(targetRenderer);
      this.setStatus(`Switched to ${this.rendererType.toUpperCase()}. ${wasRunning ? 'Restarting...' : 'Ready.'}`, this.statusEl);

      // Update mosaic info visibility based on new renderer
      this.updateMosaicInfo();

      // Restart previous source if it was running
      if (wasRunning) {
        if (this.activeInputSource === 'camera') {
          this.startCamera();
        } else if (this.activeInputSource === 'video-file') {
          // Restarting video playback after renderer toggle might be tricky if not at the same timestamp
          // For now, just re-initiate play state
          if (this.isVideoLoaded) {
            this.playVideo();
          }
        }
      }
    } catch (err) {
      this.setStatus(`Error switching renderer: ${err.message}`, this.statusEl);
      console.error(err);
    }
  }

  switchInputSource(newSource) {
    if (this.activeInputSource === newSource) {
      return; // Already on this source
    }

    // Stop current source if active
    if (this.activeInputSource === 'camera' && this.isCameraRunning) {
      this.stopCamera();
    } else if (this.activeInputSource === 'video-file' && this.isVideoPlaying) {
      this.pauseVideo();
    }

    // Update active source
    this.activeInputSource = newSource;

    // Update tab button visual state
    this.inputSourceTabButtons.forEach(button => {
      if (button.dataset.tab === newSource) {
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });

    // Update tab content visibility
    this.inputSourceTabContents.forEach(content => {
      if (content.id === `tab-${newSource}`) {
        content.classList.add('active');
      } else {
        content.classList.remove('active');
      }
    });

    // Set the current source video element for the renderer
    if (newSource === 'camera') {
      this.currentSourceVideo = this.cameraVideoElement;
      this.setStatus("Ready. Click 'Start Camera' to begin.", this.statusEl);
      // Clear player status when switching to camera
      this.playerStatusEl.textContent = "";
      // Disable player controls
      this.playPauseBtn.disabled = true;
      this.seekSlider.disabled = true;
      this.timeDisplay.textContent = "0:00 / 0:00";
    } else if (newSource === 'video-file') {
      this.currentSourceVideo = this.videoElement;
      this.setStatus("Ready. Choose a video file.", this.playerStatusEl);
      // Clear camera status when switching to video file
      this.statusEl.textContent = "";
      // Disable camera controls
      this.startBtn.disabled = true;
      this.stopBtn.disabled = true;
      this.screenshotBtn.disabled = true;
    }

    // Re-initialize renderer pipeline for the new source if available
    // Also clear canvas if no valid source to prevent displaying old frame
    if (this.renderer) {
      if (this.currentSourceVideo && this.currentSourceVideo.videoWidth > 0 && this.currentSourceVideo.videoHeight > 0) {
        this.renderer.setupPipeline(this.currentSourceVideo, this.dotSize);
      } else {
        // Clear canvas if no active video source
        this.canvas.width = this.canvas.width; // Clear by re-setting width
      }
    }
  }

  loadLocalVideo(file) {
    // Revoke previous blob URL to prevent memory leak
    if (this.currentBlobUrl) {
      URL.revokeObjectURL(this.currentBlobUrl);
    }

    this.setStatus("Loading video...", this.playerStatusEl);

    // Create blob URL and set as video source
    this.currentBlobUrl = URL.createObjectURL(file);
    this.videoElement.src = this.currentBlobUrl;
    this.videoElement.loop = true;

    // Update UI with filename
    this.currentFileDisplay.textContent = file.name;
  }

  async onVideoLoaded() {
    console.log("Video loaded:", {
      width: this.videoElement.videoWidth,
      height: this.videoElement.videoHeight,
      duration: this.videoElement.duration
    });

    this.canvas.width = this.videoElement.videoWidth;
    this.canvas.height = this.videoElement.videoHeight;

    this.resolutionValue.textContent = `${this.videoElement.videoWidth}x${this.videoElement.videoHeight}`;

    try {
      // Setup the pipeline for the video element
      if (this.renderer) {
        await this.renderer.setupPipeline(this.videoElement, this.dotSize);
      }
      this.isVideoLoaded = true;
      this.playPauseBtn.disabled = false;
      this.seekSlider.disabled = false;
      this.setStatus("Video loaded. Click Play to start.", this.playerStatusEl);
    } catch (error) {
      this.setStatus(`Failed to initialize renderer for video: ${error.message}`, this.playerStatusEl);
      console.error("Renderer init error:", error);
    }
  }

  togglePlayPause() {
    if (!this.isVideoLoaded) return;

    if (this.isVideoPlaying) {
      this.pauseVideo();
    } else {
      this.playVideo();
    }
  }

  playVideo() {
    this.videoElement.play();
    this.isVideoPlaying = true;
    this.playPauseBtn.textContent = "Pause";
    this.startVideoRenderLoop();
    this.setStatus("Playing...", this.playerStatusEl);
  }

  pauseVideo() {
    this.videoElement.pause();
    this.isVideoPlaying = false;
    this.playPauseBtn.textContent = "Play";
    if (this.videoAnimationFrame) {
      cancelAnimationFrame(this.videoAnimationFrame);
      this.videoAnimationFrame = null;
    }
    this.setStatus("Paused", this.playerStatusEl);
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
    this.isVideoPlaying = false;
    this.playPauseBtn.textContent = "Play";
    if (this.videoAnimationFrame) {
      cancelAnimationFrame(this.videoAnimationFrame);
      this.videoAnimationFrame = null;
    }
    this.setStatus("Playback ended", this.playerStatusEl);
  }

  startVideoRenderLoop() {
    const render = () => {
      if (!this.isVideoPlaying || this.activeInputSource !== 'video-file') return;

      const now = performance.now();
      const frameTime = now - this.lastFrameTime;
      this.lastFrameTime = now;

      try {
        this.renderFrame();
        this.updateFPS(frameTime);
      } catch (error) {
        console.error("Render error:", error);
        this.setStatus(`Render error: ${error.message}`, this.playerStatusEl);
        this.pauseVideo();
        return;
      }

      this.videoAnimationFrame = requestAnimationFrame(render);
    };

    // Need to reset FPS tracking for video as well
    this.lastFrameTime = performance.now();
    render();
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
      const { PortraitSegmentation } = await import("/lib/ml-inference.js");
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
      this.segmentationLoadingText.textContent = "Error: Failed to load model.";
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
    // Only start camera if current input source is 'camera'
    if (this.activeInputSource !== 'camera') {
      this.setStatus("Cannot start camera: Not in camera mode.", this.statusEl);
      return;
    }

    try {
      this.setStatus("Requesting camera access...", this.statusEl);

      // Request camera at 1080p
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          facingMode: "user",
        },
      });

      this.cameraVideoElement.srcObject = this.stream;

      // Wait for video metadata and ensure it's playing
      await new Promise((resolve, reject) => {
        this.cameraVideoElement.onloadedmetadata = async () => {
          try {
            await this.cameraVideoElement.play();
            console.log("Camera playing");
            resolve();
          } catch (err) {
            reject(err);
          }
        };
      });

      // Setup canvas size (always match video dimensions, rotation is handled by CSS)
      this.canvas.width = this.cameraVideoElement.videoWidth;
      this.canvas.height = this.cameraVideoElement.videoHeight;

      console.log(`Camera dimensions: ${this.cameraVideoElement.videoWidth}x${this.cameraVideoElement.videoHeight}`);

      // Update resolution display
      this.resolutionValue.textContent = `${this.cameraVideoElement.videoWidth}x${this.cameraVideoElement.videoHeight}`;

      // Set current source for renderer
      this.currentSourceVideo = this.cameraVideoElement;

      // Initialize renderer-specific resources
      if (this.rendererType === "webgpu") {
        await this.renderer.setupPipeline(this.currentSourceVideo, this.dotSize);
      }
      // WebGL doesn't need additional setup after init

      // Update UI
      this.startBtn.disabled = true;
      this.stopBtn.disabled = false;
      this.screenshotBtn.disabled = false;
      this.isCameraRunning = true;

      this.setStatus(`Camera running with ${this.rendererType.toUpperCase()}.`, this.statusEl);

      // Start render loop
      this.startCameraRenderLoop();
    } catch (err) {
      this.setStatus(`Camera Error: ${err.message}`, this.statusEl);
      console.error(err);
    }
  }

  stopCamera() {
    this.isCameraRunning = false;

    if (this.cameraAnimationFrame) {
      cancelAnimationFrame(this.cameraAnimationFrame);
      this.cameraAnimationFrame = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    this.startBtn.disabled = false;
    this.stopBtn.disabled = true;
    this.screenshotBtn.disabled = true;
    this.resolutionValue.textContent = "-";
    this.setStatus("Camera stopped.", this.statusEl);
  }

  startCameraRenderLoop() {
    const loop = () => {
      if (!this.isCameraRunning || this.activeInputSource !== 'camera') return;

      const frameStart = performance.now();

      try {
        this.renderFrame(); // Call the generic renderFrame
        const frameEnd = performance.now();
        const latency = frameEnd - frameStart;
        this.frameLatencies.push(latency);
        this.updateFPS();
      } catch (err) {
        console.error("Render error:", err);
        this.setStatus(`Render Error: ${err.message}`, this.statusEl);
        this.stopCamera();
        return;
      }

      this.cameraAnimationFrame = requestAnimationFrame(loop);
    };

    this.lastFpsTime = performance.now();
    this.lastLatencyUpdate = performance.now();
    this.frameLatencies = [];
    loop();
  }

  renderFrame() {
    const sourceVideo = this.currentSourceVideo;
    if (!sourceVideo) return;
    if (sourceVideo.videoWidth === 0 || sourceVideo.videoHeight === 0) return;

    switch (this.currentEffect) {
      case "halftone":
        this.renderHalftone(sourceVideo);
        break;
      case "clustering":
        this.renderClustering(sourceVideo);
        break;
      case "edges":
        this.renderEdges(sourceVideo);
        break;
      case "mosaic":
        this.renderMosaic(sourceVideo);
        break;
      case "chromatic":
        this.renderChromatic(sourceVideo);
        break;
      case "glitch":
        this.renderGlitch(sourceVideo);
        break;
      case "thermal":
        this.renderThermal(sourceVideo);
        break;
      case "pixelsort":
        this.renderPixelSort(sourceVideo);
        break;
      case "kaleidoscope":
        this.renderKaleidoscope(sourceVideo);
        break;
      case "segmentation":
        this.renderSegmentation(sourceVideo);
        break;
      default:
        this.renderPassthrough(sourceVideo);
    }
  }

  renderHalftone(sourceVideo) {
    if (this.rendererType === "webgpu") {
      this.renderer.renderHalftone(sourceVideo, this.useRandomColors);
    } else {
      this.renderer.renderHalftone(sourceVideo, this.dotSize, this.useRandomColors);
    }
  }

  renderClustering(sourceVideo) {
    // Compute algorithm string based on base algorithm and true colors toggle
    const algorithmString = this.useTrueColors
      ? `${this.clusteringAlgorithm}-true`
      : this.clusteringAlgorithm;

    this.renderer.renderClustering(
      sourceVideo,
      algorithmString,
      this.colorCount,
      this.colorThreshold
    );
  }

  renderEdges(sourceVideo) {
    // Parse edge color from hex to RGB
    const rgb = hexToRGB(this.edgeColorValue);

    this.renderer.renderEdges(
      sourceVideo,
      this.edgeAlgorithmValue,
      this.edgeThresholdValue_state,
      this.edgeOverlayValue,
      this.edgeInvertValue,
      rgb,
      this.edgeThicknessValue_state
    );
  }

  renderMosaic(sourceVideo) {
    this.renderer.renderMosaic(
      sourceVideo,
      this.mosaicBlockSizeValue_state,
      this.mosaicModeValue
    );
  }

  renderChromatic(sourceVideo) {
    // Convert center percentage to 0-1 range
    const centerX = this.chromaticCenterXValue_state / 100;
    const centerY = this.chromaticCenterYValue_state / 100;

    this.renderer.renderChromatic(
      sourceVideo,
      this.chromaticIntensityValue_state,
      this.chromaticModeValue,
      centerX,
      centerY
    );
  }

  renderGlitch(sourceVideo) {
    this.renderer.renderGlitch(
      sourceVideo,
      this.glitchModeValue,
      this.glitchIntensityValue_state,
      this.glitchBlockSizeValue_state,
      this.glitchColorShiftValue_state,
      this.glitchNoiseValue_state,
      this.glitchScanlineValue_state
    );
  }

  renderThermal(sourceVideo) {
    this.renderer.renderThermal(
      sourceVideo,
      this.thermalPaletteValue,
      this.thermalContrastValue_state,
      this.thermalInvertValue
    );
  }

  renderPixelSort(sourceVideo) {
    // Only WebGPU for now
    if (this.rendererType === "webgpu") {
      this.renderer.renderPixelSort(
        sourceVideo,
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

  renderPassthrough(sourceVideo) {
    this.renderer.renderPassthrough(sourceVideo);
  }

  updateFPS() {
    this.frameCount++;
    const now = performance.now();
    const elapsed = now - this.lastFpsTime;
    const latencyElapsed = now - this.lastLatencyUpdate;

    // Update histogram (every frame or throttled)
    // We update every frame for smoothness, the Histogram class uses a small 
    // offscreen canvas for performance.
    if (this.canvas && (this.isCameraRunning || this.isVideoPlaying)) {
      this.histogram.update(this.canvas);
    }

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

  renderKaleidoscope(sourceVideo) {
    this.renderer.renderKaleidoscope(
      sourceVideo,
      this.kaleidoscopeSegments,
      this.kaleidoscopeRotationSpeed
    );
  }

  async renderSegmentation(sourceVideo) {
    // Only render with WebGPU (segmentation not supported in WebGL yet)
    if (this.rendererType !== "webgpu") {
      // Fall back to passthrough
      this.renderPassthrough(sourceVideo);
      return;
    }

    // Check if model is loaded
    if (!this.segmentationModelLoaded || !this.segmentationML) {
      // Show passthrough while loading
      this.renderPassthrough(sourceVideo);
      return;
    }

    // Run inference with frame skipping for performance
    this.segmentationFrameCounter++;
    if (this.segmentationFrameCounter >= this.segmentationFrameSkip || !this.segmentationMask) {
      try {
        // Run segmentation inference
        const maskData = await this.segmentationML.segmentFrame(sourceVideo);

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
        sourceVideo,
        this.segmentationMode_state,
        this.segmentationBlurRadius_state,
        this.segmentationMask,
        this.segmentationSoftEdges_state,
        this.segmentationGlow_state
      );
    } else {
      // No mask yet, show passthrough
      this.renderPassthrough(sourceVideo);
    }
  }

  setStatus(text, element = this.statusEl) {
    if (element) {
      element.textContent = text;
    }
  }

  takeScreenshot() {
    if (!this.isRunning) return;

    this.canvas.toBlob((blob) => {
      if (!blob) {
        this.setStatus('Screenshot error: Failed to create image blob.');
        console.error('Screenshot error: toBlob returned null');
        return;
      }

      try {
        const now = new Date();
        const timestamp = now.toLocaleString('sv').replace(' ', '-').replace(/:/g, '');
        const filename = `cybervision-screenshot-${timestamp}.png`;

        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = filename;
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        // Clean up the object URL to avoid memory leaks
        URL.revokeObjectURL(url);

        this.setStatus(`Screenshot saved: ${filename}`);
      } catch (err) {
        this.setStatus(`Screenshot error: ${err.message}`);
        console.error('Screenshot error:', err);
      }
    }, 'image/png');
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

// Export for testing
export { CyberVision };
