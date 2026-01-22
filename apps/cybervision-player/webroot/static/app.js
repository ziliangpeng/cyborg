/* CyberVision Video Player - Main application */

import { WebGPURenderer } from "/lib/webgpu-renderer.js";
import { calculateFPS, hexToRGB } from "/lib/utils.js";

class VideoPlayer {
  constructor() {
    // Video controls
    this.videoFileInput = document.getElementById("videoFile");
    this.chooseFileBtn = document.getElementById("chooseFileBtn");
    this.dropZone = document.getElementById("dropZone");
    this.currentFileDisplay = document.getElementById("currentFile");
    this.videoElement = document.getElementById("video-element");
    this.currentBlobUrl = null;
    this.canvas = document.getElementById("video-canvas");
    this.playPauseBtn = document.getElementById("play-pause-btn");
    this.seekSlider = document.getElementById("seek-slider");
    this.timeDisplay = document.getElementById("time-display");
    this.statusEl = document.getElementById("status");

    // UI elements
    this.effectButtons = document.querySelectorAll('.effect-btn');
    this.tabButtons = document.querySelectorAll('.tab-button');
    this.tabContents = document.querySelectorAll('.tab-content');
    this.fpsValue = document.getElementById("fpsValue");
    this.latencyValue = document.getElementById("latencyValue");
    this.gpuStatus = document.getElementById("gpuStatus");
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

    // Video player state
    this.renderer = null;
    this.isPlaying = false;
    this.isVideoLoaded = false;
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

    // Kaleidoscope state
    this.segments = 8;
    this.rotationSpeed = 0.0;

    // Segmentation state
    this.segmentationML = null;
    this.segmentationModelLoaded = false;
    this.segmentationMask = null;
    this.segmentationFrameSkip = 2;
    this.segmentationFrameCounter = 0;
    this.segmentationModeValue = "blackout";
    this.segmentationBlurRadiusValue_state = 10;
    this.segmentationSoftEdgesValue = true;
    this.segmentationGlowValue = false;
    this.segmentationBackgroundImage = null;

    // FPS tracking
    this.lastFrameTime = 0;
    this.frameCount = 0;
    this.fpsUpdateInterval = 500;
    this.lastFpsUpdate = 0;

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.showStatus("Ready. Enter a video path to begin.");
    this.gpuStatus.textContent = "Not initialized";
  }

  setupEventListeners() {
    // File input change handler
    this.videoFileInput.addEventListener("change", (e) => {
      if (e.target.files[0]) {
        this.loadLocalVideo(e.target.files[0]);
      }
    });

    // Button click triggers file input
    this.chooseFileBtn.addEventListener("click", () => {
      this.videoFileInput.click();
    });

    // Drag-drop handlers
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

    // Video controls
    this.playPauseBtn.addEventListener("click", () => this.togglePlayPause());
    this.seekSlider.addEventListener("input", () => this.seek());

    // Video event listeners
    this.videoElement.addEventListener("loadedmetadata", () => this.onVideoLoaded());
    this.videoElement.addEventListener("timeupdate", () => this.updateTime());
    this.videoElement.addEventListener("ended", () => this.onVideoEnded());
    this.videoElement.addEventListener("error", (e) => this.onVideoError(e));

    // Tab switching
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
        this.effectButtons.forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        this.currentEffect = btn.dataset.effect;
        this.updateEffectControls();
      });
    });

    // Halftone controls
    this.dotSizeSlider.addEventListener("input", (e) => {
      this.dotSize = parseInt(e.target.value, 10);
      this.dotSizeValue.textContent = this.dotSize;
      if (this.renderer) {
        this.renderer.updateDotSize(this.dotSize);
      }
    });

    this.randomColorCheckbox.addEventListener("change", (e) => {
      this.useRandomColors = e.target.checked;
    });

    // Clustering controls
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

    // Edge detection controls
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

    // Mosaic controls
    this.mosaicBlockSize.addEventListener("input", (e) => {
      this.mosaicBlockSizeValue_state = parseInt(e.target.value, 10);
      this.mosaicBlockSizeValue.textContent = this.mosaicBlockSizeValue_state;
    });

    this.mosaicMode.addEventListener("change", (e) => {
      this.mosaicModeValue = e.target.value;
      this.mosaicInfo.style.display = (e.target.value === "dominant") ? "flex" : "none";
    });

    // Chromatic aberration controls
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

    // Glitch controls
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

    // Thermal controls
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

    // Pixel sort controls
    this.pixelSortAngleMode.addEventListener("change", (e) => {
      this.pixelSortAngleModeValue = e.target.value;
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
      this.pixelSortIterationsGroup.style.display = (e.target.value === "bubble") ? "block" : "none";
    });

    this.pixelSortIterations.addEventListener("input", (e) => {
      this.pixelSortIterationsValue_state = parseInt(e.target.value, 10);
      this.pixelSortIterationsValue.textContent = this.pixelSortIterationsValue_state;
    });

    // Kaleidoscope controls
    this.segmentsSlider.addEventListener("input", (e) => {
      this.segments = parseInt(e.target.value, 10);
      this.segmentsValue.textContent = this.segments;
    });

    this.rotationSpeedSlider.addEventListener("input", (e) => {
      this.rotationSpeed = parseFloat(e.target.value);
      this.rotationSpeedValue.textContent = this.rotationSpeed.toFixed(2);
    });

    // Segmentation controls
    this.segmentationMode.addEventListener("change", (e) => {
      this.segmentationModeValue = e.target.value;
      this.segmentationBlurGroup.style.display = (e.target.value === "blur") ? "block" : "none";
      this.segmentationBackgroundGroup.style.display = (e.target.value === "replace") ? "block" : "none";
    });

    this.segmentationBlurRadius.addEventListener("input", (e) => {
      this.segmentationBlurRadiusValue_state = parseInt(e.target.value, 10);
      this.segmentationBlurRadiusValue.textContent = this.segmentationBlurRadiusValue_state;
    });

    this.segmentationSoftEdges.addEventListener("change", (e) => {
      this.segmentationSoftEdgesValue = e.target.checked;
    });

    this.segmentationGlow.addEventListener("change", (e) => {
      this.segmentationGlowValue = e.target.checked;
    });

    this.segmentationBackgroundUpload?.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          this.backgroundPreviewImg.src = event.target.result;
          this.backgroundPreview.style.display = "block";
          const img = new Image();
          img.onload = () => {
            this.segmentationBackgroundImage = img;
          };
          img.src = event.target.result;
        };
        reader.readAsDataURL(file);
      }
    });
  }

  updateEffectControls() {
    // Hide all effect controls
    this.halftoneControls.style.display = "none";
    this.clusteringControls.style.display = "none";
    this.edgesControls.style.display = "none";
    this.mosaicControls.style.display = "none";
    this.chromaticControls.style.display = "none";
    this.glitchControls.style.display = "none";
    this.thermalControls.style.display = "none";
    this.pixelSortControls.style.display = "none";
    this.kaleidoscopeControls.style.display = "none";
    this.segmentationControls.style.display = "none";

    // Show controls for current effect
    switch (this.currentEffect) {
      case "halftone":
        this.halftoneControls.style.display = "block";
        break;
      case "clustering":
        this.clusteringControls.style.display = "block";
        break;
      case "edges":
        this.edgesControls.style.display = "block";
        break;
      case "mosaic":
        this.mosaicControls.style.display = "block";
        break;
      case "chromatic":
        this.chromaticControls.style.display = "block";
        break;
      case "glitch":
        this.glitchControls.style.display = "block";
        break;
      case "thermal":
        this.thermalControls.style.display = "block";
        break;
      case "pixelsort":
        this.pixelSortControls.style.display = "block";
        break;
      case "kaleidoscope":
        this.kaleidoscopeControls.style.display = "block";
        break;
      case "segmentation":
        this.segmentationControls.style.display = "block";
        break;
    }
  }

  loadLocalVideo(file) {
    // Revoke previous blob URL to prevent memory leak
    if (this.currentBlobUrl) {
      URL.revokeObjectURL(this.currentBlobUrl);
    }

    this.showStatus("Loading video...");

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
      await this.initRenderer();
      this.isVideoLoaded = true;
      this.playPauseBtn.disabled = false;
      this.seekSlider.disabled = false;
      this.showStatus("Video loaded. Click Play to start.");
    } catch (error) {
      this.showStatus(`Failed to initialize renderer: ${error.message}`);
      console.error("Renderer init error:", error);
    }
  }

  async initRenderer() {
    this.renderer = new WebGPURenderer();
    await this.renderer.init(this.canvas);
    await this.renderer.setupPipeline(this.videoElement, this.dotSize);
    this.gpuStatus.textContent = "WebGPU";
    console.log("Using WebGPU renderer");
  }

  async loadSegmentationModel() {
    if (this.segmentationModelLoaded || this.segmentationML) return;

    try {
      this.segmentationLoadingText.textContent = "Loading ML model...";
      const { PortraitSegmentation } = await import("/libs/cybervision-core/ml-inference.js");
      this.segmentationML = new PortraitSegmentation();
      await this.segmentationML.loadModel("/models/segmentation.onnx");
      this.segmentationModelLoaded = true;
      this.segmentationLoading.style.display = "none";
      console.log("Segmentation model loaded successfully");
    } catch (error) {
      this.segmentationModelLoaded = false;
      this.segmentationML = null;
      this.segmentationLoadingText.textContent = "Model not found. See webroot/models/DOWNLOAD_MODEL.md";
      console.warn("Segmentation model not available:", error.message);
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
    this.showStatus(`Video error: ${message}`);
  }

  startRenderLoop() {
    const render = () => {
      if (!this.isPlaying) return;

      const now = performance.now();
      const frameTime = now - this.lastFrameTime;
      this.lastFrameTime = now;

      try {
        this.renderFrame();
        this.updateFPS(frameTime);
      } catch (error) {
        console.error("Render error:", error);
        this.showStatus(`Render error: ${error.message}`);
        this.pause();
        return;
      }

      this.animationFrame = requestAnimationFrame(render);
    };

    this.lastFrameTime = performance.now();
    render();
  }

  updateFPS(frameTime) {
    this.frameCount++;
    const now = performance.now();

    if (now - this.lastFpsUpdate >= this.fpsUpdateInterval) {
      const fps = Math.round(this.frameCount / ((now - this.lastFpsUpdate) / 1000));
      this.fpsValue.textContent = fps;
      this.latencyValue.textContent = Math.round(frameTime) + " ms";
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }
  }

  async renderFrame() {
    if (!this.renderer || !this.videoElement.videoWidth) return;

    switch (this.currentEffect) {
      case "original":
        this.renderer.renderPassthrough(this.videoElement);
        break;

      case "halftone":
        this.renderer.renderHalftone(this.videoElement, this.useRandomColors);
        break;

      case "clustering":
        this.renderer.renderClustering(
          this.videoElement,
          this.clusteringAlgorithm,
          this.colorCount,
          this.colorThreshold
        );
        break;

      case "edges":
        const edgeRGB = hexToRGB(this.edgeColorValue);
        this.renderer.renderEdges(
          this.videoElement,
          this.edgeAlgorithmValue,
          this.edgeThresholdValue_state,
          this.edgeOverlayValue,
          this.edgeInvertValue,
          edgeRGB,
          this.edgeThicknessValue_state
        );
        break;

      case "mosaic":
        this.renderer.renderMosaic(
          this.videoElement,
          this.mosaicBlockSizeValue_state,
          this.mosaicModeValue
        );
        break;

      case "chromatic":
        const intensityNormalized = this.chromaticIntensityValue_state / 1000.0;
        const centerX = this.chromaticCenterXValue_state / 100.0;
        const centerY = this.chromaticCenterYValue_state / 100.0;
        this.renderer.renderChromatic(
          this.videoElement,
          this.chromaticModeValue,
          intensityNormalized,
          centerX,
          centerY
        );
        break;

      case "glitch":
        const glitchIntensityNorm = this.glitchIntensityValue_state / 100.0;
        this.renderer.renderGlitch(
          this.videoElement,
          this.glitchModeValue,
          glitchIntensityNorm,
          this.glitchBlockSizeValue_state,
          this.glitchColorShiftValue_state / 100.0,
          this.glitchNoiseValue_state,
          this.glitchScanlineValue_state
        );
        break;

      case "thermal":
        this.renderer.renderThermal(
          this.videoElement,
          this.thermalPaletteValue,
          this.thermalContrastValue_state,
          this.thermalInvertValue
        );
        break;

      case "pixelsort":
        const direction = this.pixelSortAngleModeValue === "preset"
          ? this.pixelSortDirectionValue
          : "horizontal";
        const angle = this.pixelSortAngleModeValue === "rotate"
          ? this.pixelSortAngleValue_state
          : 0;
        const iterations = this.pixelSortAlgorithmValue === "bubble"
          ? this.pixelSortIterationsValue_state
          : 1;
        this.renderer.renderPixelSort(
          this.videoElement,
          direction,
          angle,
          this.pixelSortThresholdModeValue,
          this.pixelSortThresholdLowValue_state,
          this.pixelSortThresholdHighValue_state,
          this.pixelSortKeyValue,
          this.pixelSortOrderValue,
          this.pixelSortAlgorithmValue,
          iterations
        );
        break;

      case "kaleidoscope":
        this.renderer.renderKaleidoscope(this.videoElement, this.segments, this.rotationSpeed);
        break;

      case "segmentation":
        await this.renderSegmentation();
        break;

      default:
        this.renderer.renderPassthrough(this.videoElement);
    }
  }

  async renderSegmentation() {
    if (!this.segmentationModelLoaded && !this.segmentationML) {
      await this.loadSegmentationModel();
    }

    if (!this.segmentationModelLoaded || !this.segmentationML) {
      this.renderer.renderPassthrough(this.videoElement);
      return;
    }

    this.segmentationFrameCounter++;
    if (this.segmentationFrameCounter >= this.segmentationFrameSkip || !this.segmentationMask) {
      const maskData = await this.segmentationML.segmentFrame(this.videoElement);
      this.segmentationMask = this.segmentationML.postprocessMask(maskData);
      this.segmentationFrameCounter = 0;
    }

    if (this.segmentationMask) {
      this.renderer.renderSegmentation(
        this.videoElement,
        this.segmentationModeValue,
        this.segmentationBlurRadiusValue_state,
        this.segmentationMask,
        this.segmentationSoftEdgesValue,
        this.segmentationGlowValue,
        this.segmentationBackgroundImage
      );
    } else {
      this.renderer.renderPassthrough(this.videoElement);
    }
  }

  showStatus(message) {
    this.statusEl.textContent = message;
  }
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const player = new VideoPlayer();
});

// Export for testing
export { VideoPlayer };
