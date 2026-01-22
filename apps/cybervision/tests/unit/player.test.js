import { describe, it, expect, beforeEach, afterEach } from 'vitest';

// Mock dependencies are configured via vitest.config.js aliases

// Helper function to create a mock video element
function createMockVideoElement() {
  return {
    currentTime: 0,
    duration: 100,
    paused: true,
    play: vi.fn().mockResolvedValue(undefined),
    pause: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  };
}

// Helper function to create a mock DOM
function setupMockDOM() {
  document.body.innerHTML = `
    <!-- Video controls -->
    <input type="file" id="videoFile" accept="video/*" style="display: none">
    <div id="dropZone" class="drop-zone">
      <button id="chooseFileBtn">Choose Video File</button>
    </div>
    <div id="currentFile"></div>
    <video id="video-element"></video>
    <video id="camera-video-element"></video>
    <canvas id="video-canvas"></canvas>
    <canvas id="canvas"></canvas>
    <button id="play-pause-btn">Play</button>
    <button id="startBtn">Start Camera</button>
    <button id="stopBtn">Stop Camera</button>
    <button id="screenshotBtn">Screenshot</button>
    <input id="seek-slider" type="range" min="0" max="100" value="0" />
    <div id="time-display">0:00 / 0:00</div>
    <div id="status-message"></div>
    <div id="player-status"></div>

    <!-- Core UI elements -->
    <div id="status"></div>
    <div id="fpsValue">0</div>
    <div id="latencyValue">0</div>
    <div id="gpuStatus">Unknown</div>
    <input type="checkbox" id="webglToggle" />
    <div id="resolutionValue">-</div>

    <!-- Input source tabs -->
    <button class="input-source-tab-button" data-tab="camera">Camera</button>
    <button class="input-source-tab-button" data-tab="video-file">Video File</button>
    <div class="input-source-tab-content" id="tab-camera"></div>
    <div class="input-source-tab-content" id="tab-video-file"></div>

    <!-- Effect tab structure -->
    <div class="effect-tab-button" data-tab="effects">Effects</div>
    <div class="effect-tab-button" data-tab="settings">Settings</div>
    <div class="effect-tab-content" id="tab-effects"></div>
    <div class="effect-tab-content" id="tab-settings"></div>

    <!-- Effect buttons -->
    <button class="effect-btn" data-effect="original">Original</button>
    <button class="effect-btn" data-effect="halftone">Halftone</button>
    <button class="effect-btn" data-effect="duotone">Duotone</button>
    <button class="effect-btn" data-effect="dither">Dither</button>
    <button class="effect-btn" data-effect="posterize">Posterize</button>
    <button class="effect-btn" data-effect="clustering">Clustering</button>
    <button class="effect-btn" data-effect="edges">Edges</button>
    <button class="effect-btn" data-effect="twirl">Twirl</button>
    <button class="effect-btn" data-effect="vignette">Vignette</button>
    <button class="effect-btn" data-effect="mosaic">Mosaic</button>
    <button class="effect-btn" data-effect="chromatic">Chromatic</button>
    <button class="effect-btn" data-effect="glitch">Glitch</button>
    <button class="effect-btn" data-effect="thermal">Thermal</button>
    <button class="effect-btn" data-effect="pixel-sort">Pixel Sort</button>
    <button class="effect-btn" data-effect="kaleidoscope">Kaleidoscope</button>
    <button class="effect-btn" data-effect="segmentation">Segmentation</button>

    <!-- Halftone controls -->
    <div id="halftoneControls" class="effect-controls">
      <input id="dotSizeSlider" type="range" min="2" max="20" value="8" />
      <span id="dotSizeValue">8</span>
      <input id="randomColorCheckbox" type="checkbox" />
    </div>

    <!-- Duotone controls -->
    <div id="duotoneControls" class="effect-controls">
      <input id="duotoneShadow" type="color" value="#1b1f2a" />
      <input id="duotoneHighlight" type="color" value="#f2c14e" />
    </div>

    <!-- Dither controls -->
    <div id="ditherControls" class="effect-controls">
      <input id="ditherScale" type="range" min="1" max="8" value="2" />
      <span id="ditherScaleValue">2</span>
      <input id="ditherLevels" type="range" min="2" max="8" value="4" />
      <span id="ditherLevelsValue">4</span>
    </div>

    <!-- Posterize controls -->
    <div id="posterizeControls" class="effect-controls">
      <input id="posterizeLevels" type="range" min="2" max="8" value="4" />
      <span id="posterizeLevelsValue">4</span>
    </div>

    <!-- Clustering controls -->
    <div id="clusteringControls" class="effect-controls">
      <select id="algorithmSelect">
        <option value="quantization-kmeans">K-Means</option>
      </select>
      <input id="trueColorsCheckbox" type="checkbox" />
      <input id="colorCountSlider" type="range" min="2" max="32" value="8" />
      <span id="colorCountValue">8</span>
      <input id="thresholdSlider" type="range" min="0" max="1" step="0.01" value="0.1" />
      <span id="thresholdValue">0.1</span>
    </div>

    <!-- Edge detection controls -->
    <div id="edgesControls" class="effect-controls">
      <select id="edgeAlgorithm">
        <option value="sobel">Sobel</option>
      </select>
      <input id="edgeThreshold" type="range" min="0" max="1" step="0.01" value="0.1" />
      <span id="edgeThresholdValue">0.1</span>
      <input id="edgeOverlay" type="checkbox" />
      <input id="edgeInvert" type="checkbox" />
      <input id="edgeColor" type="color" value="#ffffff" />
      <input id="edgeThickness" type="range" min="1" max="5" value="1" />
      <span id="edgeThicknessValue">1</span>
    </div>

    <!-- Twirl controls -->
    <div id="twirlControls" class="effect-controls">
      <input id="twirlStrength" type="range" min="-4" max="4" value="0" />
      <span id="twirlStrengthValue">0.0</span>
      <input id="twirlRadius" type="range" min="0.1" max="1.0" value="0.5" />
      <span id="twirlRadiusValue">0.5</span>
    </div>

    <!-- Vignette controls -->
    <div id="vignetteControls" class="effect-controls">
      <input id="vignetteStrength" type="range" min="0" max="1" value="0.5" />
      <span id="vignetteStrengthValue">0.5</span>
      <input id="vignetteGrain" type="range" min="0" max="0.3" value="0.08" />
      <span id="vignetteGrainValue">0.08</span>
    </div>

    <!-- Mosaic controls -->
    <div id="mosaicControls" class="effect-controls">
      <input id="mosaicBlockSize" type="range" min="4" max="32" value="8" />
      <span id="mosaicBlockSizeValue">8</span>
      <select id="mosaicMode">
        <option value="center">Center</option>
      </select>
      <div id="mosaicInfo"></div>
    </div>

    <!-- Chromatic aberration controls -->
    <div id="chromaticControls" class="effect-controls">
      <input id="chromaticIntensity" type="range" min="0" max="50" value="10" />
      <span id="chromaticIntensityValue">10</span>
      <select id="chromaticMode">
        <option value="radial">Radial</option>
      </select>
      <input id="chromaticCenterX" type="range" min="0" max="100" value="50" />
      <span id="chromaticCenterXValue">50</span>
      <input id="chromaticCenterY" type="range" min="0" max="100" value="50" />
      <span id="chromaticCenterYValue">50</span>
    </div>

    <!-- Glitch controls -->
    <div id="glitchControls" class="effect-controls">
      <select id="glitchMode">
        <option value="slices">Slices</option>
      </select>
      <input id="glitchIntensity" type="range" min="0" max="50" value="12" />
      <span id="glitchIntensityValue">12</span>
      <input id="glitchBlockSize" type="range" min="4" max="64" value="24" />
      <span id="glitchBlockSizeValue">24</span>
      <input id="glitchColorShift" type="range" min="0" max="20" value="4" />
      <span id="glitchColorShiftValue">4</span>
      <input id="glitchNoise" type="range" min="0" max="1" step="0.01" value="0.15" />
      <span id="glitchNoiseValue">0.15</span>
      <input id="glitchScanline" type="range" min="0" max="1" step="0.01" value="0.3" />
      <span id="glitchScanlineValue">0.3</span>
    </div>

    <!-- Thermal controls -->
    <div id="thermalControls" class="effect-controls">
      <select id="thermalPalette">
        <option value="classic">Classic</option>
      </select>
      <input id="thermalContrast" type="range" min="0.5" max="2" step="0.1" value="1" />
      <span id="thermalContrastValue">1.0</span>
      <input id="thermalInvert" type="checkbox" />
    </div>

    <!-- Pixel sort controls -->
    <div id="pixelSortControls" class="effect-controls">
      <select id="pixelSortAngleMode">
        <option value="preset">Preset</option>
        <option value="custom">Custom</option>
      </select>
      <select id="pixelSortDirection">
        <option value="horizontal">Horizontal</option>
      </select>
      <div id="pixelSortDirectionGroup"></div>
      <input id="pixelSortAngle" type="range" min="0" max="360" value="0" />
      <span id="pixelSortAngleValue">0</span>
      <div id="pixelSortAngleGroup"></div>
      <select id="pixelSortThresholdMode">
        <option value="brightness">Brightness</option>
        <option value="saturation">Saturation</option>
        <option value="hue">Hue</option>
        <option value="edge">Edge</option>
      </select>
      <input id="pixelSortThresholdLow" type="range" min="0" max="1" step="0.01" value="0.25" />
      <span id="pixelSortThresholdLowValue">0.25</span>
      <input id="pixelSortThresholdHigh" type="range" min="0" max="1" step="0.01" value="0.75" />
      <span id="pixelSortThresholdHighValue">0.75</span>
      <select id="pixelSortKey">
        <option value="luminance">Luminance</option>
        <option value="hue">Hue</option>
        <option value="saturation">Saturation</option>
        <option value="red">Red Channel</option>
        <option value="green">Green Channel</option>
        <option value="blue">Blue Channel</option>
      </select>
      <select id="pixelSortOrder">
        <option value="ascending">Ascending</option>
        <option value="descending">Descending</option>
      </select>
      <select id="pixelSortAlgorithm">
        <option value="bitonic">Bitonic</option>
        <option value="bubble">Bubble (Stylized)</option>
      </select>
      <input id="pixelSortIterations" type="range" min="1" max="100" value="50" />
      <span id="pixelSortIterationsValue">50</span>
      <div id="pixelSortIterationsGroup"></div>
    </div>

    <!-- Kaleidoscope controls -->
    <div id="kaleidoscopeControls" class="effect-controls">
      <input id="segmentsSlider" type="range" min="2" max="16" value="8" />
      <span id="segmentsValue">8</span>
      <input id="rotationSpeedSlider" type="range" min="0" max="5" step="0.1" value="0" />
      <span id="rotationSpeedValue">0</span>
    </div>

    <!-- Segmentation controls -->
    <div id="segmentationControls" class="effect-controls">
      <select id="segmentationMode">
        <option value="blur">Blur</option>
        <option value="blackout">Blackout</option>
        <option value="replace">Replace</option>
      </select>
      <input id="segmentationBlurRadius" type="range" min="1" max="30" value="10" />
      <span id="segmentationBlurRadiusValue">10</span>
      <div id="segmentationBlurGroup"></div>
      <div id="segmentationBackgroundGroup">
        <input id="segmentationBackgroundUpload" type="file" accept="image/*" />
        <div id="backgroundPreview">
          <img id="backgroundPreviewImg" />
        </div>
      </div>
      <div id="segmentationLoading">
        <span id="segmentationLoadingText">Loading model...</span>
      </div>
      <input id="segmentationSoftEdges" type="checkbox" checked />
      <input id="segmentationGlow" type="checkbox" />
    </div>
  `;
}

async function createTestPlayer() {
  const { CyberVision } = await import('../../webroot/static/app.js');
  class TestCyberVision extends CyberVision {
    async init() {}
  }
  return new TestCyberVision();
}

function setMediaElementProperty(element, property, value) {
  Object.defineProperty(element, property, {
    value,
    writable: true,
    configurable: true,
  });
}

describe('CyberVision - Time Formatting', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should format time correctly for valid seconds', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.formatTime(0)).toBe('0:00');
    expect(player.formatTime(30)).toBe('0:30');
    expect(player.formatTime(60)).toBe('1:00');
    expect(player.formatTime(90)).toBe('1:30');
    expect(player.formatTime(125)).toBe('2:05');
    expect(player.formatTime(3661)).toBe('61:01');
  });

  it('should handle NaN input', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.formatTime(NaN)).toBe('0:00');
  });

  it('should pad seconds with leading zero', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.formatTime(5)).toBe('0:05');
    expect(player.formatTime(65)).toBe('1:05');
  });
});

describe('CyberVision - Effect Parameters', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should initialize with default effect parameters', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.dotSize).toBe(8);
    expect(player.useRandomColors).toBe(false);
    expect(player.colorCount).toBe(8);
    expect(player.kaleidoscopeSegments).toBe(8);
    expect(player.kaleidoscopeRotationSpeed).toBe(0.0);
    expect(player.currentEffect).toBe('segmentation');
  });

  it('should initialize video player state', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.isVideoPlaying).toBe(false);
    expect(player.isVideoLoaded).toBe(false);
    expect(player.renderer).toBe(null);
    expect(player.videoAnimationFrame).toBe(null);
  });

  it('should initialize segmentation state variables correctly', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    expect(player.segmentationML).toBe(null);
    expect(player.segmentationModelLoaded).toBe(false);
    expect(player.segmentationMode_state).toBe('blackout');
    expect(player.segmentationBlurRadius_state).toBe(10);
  });
});

describe('CyberVision - Local Video Loading', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should load local video file and create blob URL', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    const mockFile = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    player.loadLocalVideo(mockFile);

    expect(player.videoElement.src).toContain('blob:');
    expect(document.getElementById('currentFile').textContent).toBe('test.mp4');
  });

  it('should revoke previous blob URL when loading new video', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL');

    const file1 = new File(['video1'], 'video1.mp4', { type: 'video/mp4' });
    const file2 = new File(['video2'], 'video2.mp4', { type: 'video/mp4' });

    player.loadLocalVideo(file1);
    const firstBlobUrl = player.currentBlobUrl;

    player.loadLocalVideo(file2);

    expect(revokeObjectURL).toHaveBeenCalledWith(firstBlobUrl);
  });

  it('should update current file display with filename', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    const mockFile = new File(['content'], 'my-video.mov', { type: 'video/quicktime' });
    player.loadLocalVideo(mockFile);

    expect(player.currentFileDisplay.textContent).toBe('my-video.mov');
  });

  it('should show loading status when loading video', async () => {
    const { CyberVision } = await import('../../webroot/static/app.js');
    const player = new CyberVision();

    const mockFile = new File(['content'], 'test.mp4', { type: 'video/mp4' });
    player.loadLocalVideo(mockFile);

    const status = document.getElementById('player-status');
    expect(status.textContent).toContain('Loading video');
  });
});

describe('CyberVision - Playback Controls', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('updates time display and seek slider when video is loaded', async () => {
    const player = await createTestPlayer();
    player.isVideoLoaded = true;

    setMediaElementProperty(player.videoElement, 'currentTime', 30);
    setMediaElementProperty(player.videoElement, 'duration', 120);

    player.updateTime();

    expect(player.timeDisplay.textContent).toBe('0:30 / 2:00');
    expect(Number(player.seekSlider.value)).toBeCloseTo(25);
  });

  it('does not update time display when video is not loaded', async () => {
    const player = await createTestPlayer();
    player.isVideoLoaded = false;

    setMediaElementProperty(player.videoElement, 'currentTime', 50);
    setMediaElementProperty(player.videoElement, 'duration', 100);
    player.timeDisplay.textContent = '0:00 / 0:00';
    player.seekSlider.value = '0';

    player.updateTime();

    expect(player.timeDisplay.textContent).toBe('0:00 / 0:00');
    expect(player.seekSlider.value).toBe('0');
  });

  it('seeks to the correct position based on the slider value', async () => {
    const player = await createTestPlayer();
    player.isVideoLoaded = true;

    setMediaElementProperty(player.videoElement, 'duration', 200);
    setMediaElementProperty(player.videoElement, 'currentTime', 0);
    player.seekSlider.value = '50';

    player.seek();

    expect(player.videoElement.currentTime).toBe(100);
  });

  it('toggles play and pause based on current state', async () => {
    const player = await createTestPlayer();
    const playSpy = vi.spyOn(player, 'playVideo').mockImplementation(() => {});
    const pauseSpy = vi.spyOn(player, 'pauseVideo').mockImplementation(() => {});

    player.isVideoLoaded = true;
    player.isVideoPlaying = false;
    player.togglePlayPause();

    expect(playSpy).toHaveBeenCalledTimes(1);
    expect(pauseSpy).not.toHaveBeenCalled();

    player.isVideoPlaying = true;
    player.togglePlayPause();

    expect(pauseSpy).toHaveBeenCalledTimes(1);
  });
});

describe('CyberVision - Effect Control Visibility', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('toggles low-effort effect controls', async () => {
    const player = await createTestPlayer();

    player.currentEffect = 'duotone';
    player.updateEffectControls();
    expect(player.duotoneControls.style.display).toBe('block');
    expect(player.ditherControls.style.display).toBe('none');
    expect(player.posterizeControls.style.display).toBe('none');
    expect(player.twirlControls.style.display).toBe('none');
    expect(player.vignetteControls.style.display).toBe('none');

    player.currentEffect = 'dither';
    player.updateEffectControls();
    expect(player.ditherControls.style.display).toBe('block');
    expect(player.duotoneControls.style.display).toBe('none');
    expect(player.posterizeControls.style.display).toBe('none');

    player.currentEffect = 'posterize';
    player.updateEffectControls();
    expect(player.posterizeControls.style.display).toBe('block');
    expect(player.ditherControls.style.display).toBe('none');

    player.currentEffect = 'twirl';
    player.updateEffectControls();
    expect(player.twirlControls.style.display).toBe('block');
    expect(player.edgesControls.style.display).toBe('none');

    player.currentEffect = 'vignette';
    player.updateEffectControls();
    expect(player.vignetteControls.style.display).toBe('block');
    expect(player.twirlControls.style.display).toBe('none');
  });

  it('shows mosaic info only for dominant mode in WebGL', async () => {
    const player = await createTestPlayer();
    player.rendererType = 'webgl';
    player.mosaicModeValue = 'dominant';

    player.updateMosaicInfo();
    expect(player.mosaicInfo.style.display).toBe('flex');

    player.mosaicModeValue = 'center';
    player.updateMosaicInfo();
    expect(player.mosaicInfo.style.display).toBe('none');

    player.mosaicModeValue = 'dominant';
    player.rendererType = 'webgpu';
    player.updateMosaicInfo();
    expect(player.mosaicInfo.style.display).toBe('none');
  });

  it('toggles pixel sort iteration controls based on algorithm', async () => {
    const player = await createTestPlayer();
    player.pixelSortAlgorithmValue = 'bubble';

    player.updatePixelSortIterationsVisibility();
    expect(player.pixelSortIterationsGroup.style.display).toBe('block');

    player.pixelSortAlgorithmValue = 'bitonic';
    player.updatePixelSortIterationsVisibility();
    expect(player.pixelSortIterationsGroup.style.display).toBe('none');
  });

  it('switches segmentation controls based on mode', async () => {
    const player = await createTestPlayer();

    player.segmentationMode_state = 'blur';
    player.updateSegmentationControlsVisibility();
    expect(player.segmentationBlurGroup.style.display).toBe('block');
    expect(player.segmentationBackgroundGroup.style.display).toBe('none');

    player.segmentationMode_state = 'replace';
    player.updateSegmentationControlsVisibility();
    expect(player.segmentationBlurGroup.style.display).toBe('none');
    expect(player.segmentationBackgroundGroup.style.display).toBe('block');

    player.segmentationMode_state = 'blackout';
    player.updateSegmentationControlsVisibility();
    expect(player.segmentationBlurGroup.style.display).toBe('none');
    expect(player.segmentationBackgroundGroup.style.display).toBe('none');
  });
});

describe('CyberVision - Renderer Parameters', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('builds clustering algorithm strings based on true color toggle', async () => {
    const player = await createTestPlayer();
    const renderer = { renderClustering: vi.fn() };
    const sourceVideo = {};

    player.renderer = renderer;
    player.clusteringAlgorithm = 'quantization-kmeans';
    player.colorCount = 6;
    player.colorThreshold = 0.2;

    player.useTrueColors = true;
    player.renderClustering(sourceVideo);

    player.useTrueColors = false;
    player.renderClustering(sourceVideo);

    expect(renderer.renderClustering).toHaveBeenNthCalledWith(
      1,
      sourceVideo,
      'quantization-kmeans-true',
      6,
      0.2
    );
    expect(renderer.renderClustering).toHaveBeenNthCalledWith(
      2,
      sourceVideo,
      'quantization-kmeans',
      6,
      0.2
    );
  });

  it('passes parsed edge color to the renderer', async () => {
    const player = await createTestPlayer();
    const renderer = { renderEdges: vi.fn() };
    const sourceVideo = {};

    player.renderer = renderer;
    player.edgeAlgorithmValue = 'sobel';
    player.edgeThresholdValue_state = 0.2;
    player.edgeOverlayValue = true;
    player.edgeInvertValue = false;
    player.edgeColorValue = '#ff0000';
    player.edgeThicknessValue_state = 3;

    player.renderEdges(sourceVideo);

    expect(renderer.renderEdges).toHaveBeenCalledWith(
      sourceVideo,
      'sobel',
      0.2,
      true,
      false,
      [1, 0, 0],
      3
    );
  });

  it('normalizes chromatic center percentages before rendering', async () => {
    const player = await createTestPlayer();
    const renderer = { renderChromatic: vi.fn() };
    const sourceVideo = {};

    player.renderer = renderer;
    player.chromaticIntensityValue_state = 12;
    player.chromaticModeValue = 'radial';
    player.chromaticCenterXValue_state = 25;
    player.chromaticCenterYValue_state = 75;

    player.renderChromatic(sourceVideo);

    expect(renderer.renderChromatic).toHaveBeenCalledWith(
      sourceVideo,
      12,
      'radial',
      0.25,
      0.75
    );
  });

  it('updates FPS and latency displays after a second elapses', async () => {
    const player = await createTestPlayer();
    const nowSpy = vi.spyOn(performance, 'now').mockReturnValue(1000);

    player.isVideoPlaying = true;
    player.frameCount = 59;
    player.lastFpsTime = 0;
    player.lastLatencyUpdate = 0;
    player.frameLatencies = [10, 20, 30];
    player.histogram.update = vi.fn();

    player.updateFPS();

    expect(player.fpsValue.textContent).toBe('60');
    expect(player.latencyValue.textContent).toBe('20.00 ms');
    expect(player.histogram.update).toHaveBeenCalledWith(player.canvas);

    nowSpy.mockRestore();
  });
});

describe('CyberVision - Event Listeners', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('switches input sources when tabs are clicked', async () => {
    const player = await createTestPlayer();
    const switchSpy = vi.spyOn(player, 'switchInputSource').mockImplementation(() => {});

    player.setupEventListeners();

    const videoTab = document.querySelector('.input-source-tab-button[data-tab="video-file"]');
    videoTab.dispatchEvent(new Event('click'));

    expect(switchSpy).toHaveBeenCalledWith('video-file');
  });

  it('loads files from input change and drop events', async () => {
    const player = await createTestPlayer();
    const loadSpy = vi.spyOn(player, 'loadLocalVideo').mockImplementation(() => {});

    player.setupEventListeners();

    const file = new File(['data'], 'clip.mp4', { type: 'video/mp4' });
    Object.defineProperty(player.videoFileInput, 'files', {
      value: [file],
      configurable: true,
    });
    player.videoFileInput.dispatchEvent(new Event('change'));
    expect(loadSpy).toHaveBeenCalledWith(file);

    const dropEvent = new Event('drop');
    Object.defineProperty(dropEvent, 'dataTransfer', {
      value: { files: [file] },
      configurable: true,
    });
    player.dropZone.dispatchEvent(dropEvent);
    expect(loadSpy).toHaveBeenCalledWith(file);
  });

  it('toggles drop zone state and opens file picker', async () => {
    const player = await createTestPlayer();
    player.videoFileInput.click = vi.fn();

    player.setupEventListeners();

    player.dropZone.dispatchEvent(new Event('dragover'));
    expect(player.dropZone.classList.contains('drag-over')).toBe(true);

    player.dropZone.dispatchEvent(new Event('dragleave'));
    expect(player.dropZone.classList.contains('drag-over')).toBe(false);

    player.chooseFileBtn.dispatchEvent(new Event('click'));
    expect(player.videoFileInput.click).toHaveBeenCalled();
  });

  it('updates effect state when selecting an effect button', async () => {
    const player = await createTestPlayer();
    const updateSpy = vi.spyOn(player, 'updateEffectControls').mockImplementation(() => {});

    player.setupEventListeners();

    const buttons = Array.from(document.querySelectorAll('.effect-btn'));
    const targetButton = buttons.find((btn) => btn.dataset.effect === 'mosaic');
    const otherButton = buttons.find((btn) => btn.dataset.effect === 'halftone');

    targetButton.dispatchEvent(new Event('click'));

    expect(player.currentEffect).toBe('mosaic');
    expect(targetButton.classList.contains('selected')).toBe(true);
    expect(targetButton.getAttribute('aria-checked')).toBe('true');
    expect(otherButton.getAttribute('aria-checked')).toBe('false');
    expect(updateSpy).toHaveBeenCalled();
  });
});

describe('CyberVision - Input Source Switching', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('switches to video-file mode and sets up pipeline', async () => {
    const player = await createTestPlayer();
    const stopSpy = vi.spyOn(player, 'stopCamera').mockImplementation(() => {});

    player.activeInputSource = 'camera';
    player.isCameraRunning = true;
    player.renderer = { setupPipeline: vi.fn() };

    setMediaElementProperty(player.videoElement, 'videoWidth', 640);
    setMediaElementProperty(player.videoElement, 'videoHeight', 360);

    player.switchInputSource('video-file');

    expect(stopSpy).toHaveBeenCalled();
    expect(player.activeInputSource).toBe('video-file');
    expect(player.currentSourceVideo).toBe(player.videoElement);
    expect(player.playerStatusEl.textContent).toContain('Ready. Choose a video file.');
    expect(player.statusEl.textContent).toBe('');
    expect(player.startBtn.disabled).toBe(true);
    expect(player.stopBtn.disabled).toBe(true);
    expect(player.screenshotBtn.disabled).toBe(true);
    expect(player.renderer.setupPipeline).toHaveBeenCalledWith(player.videoElement, player.dotSize);
  });

  it('switches to camera mode and pauses video playback', async () => {
    const player = await createTestPlayer();
    const pauseSpy = vi.spyOn(player, 'pauseVideo').mockImplementation(() => {});

    player.activeInputSource = 'video-file';
    player.isVideoPlaying = true;

    player.switchInputSource('camera');

    expect(pauseSpy).toHaveBeenCalled();
    expect(player.activeInputSource).toBe('camera');
    expect(player.currentSourceVideo).toBe(player.cameraVideoElement);
    expect(player.statusEl.textContent).toContain("Ready. Click 'Start Camera' to begin.");
    expect(player.playerStatusEl.textContent).toBe('');
    expect(player.playPauseBtn.disabled).toBe(true);
    expect(player.seekSlider.disabled).toBe(true);
    expect(player.timeDisplay.textContent).toBe('0:00 / 0:00');
  });
});

describe('CyberVision - Video Lifecycle', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes renderer state when video metadata loads', async () => {
    const player = await createTestPlayer();
    player.renderer = { setupPipeline: vi.fn().mockResolvedValue() };

    setMediaElementProperty(player.videoElement, 'videoWidth', 1280);
    setMediaElementProperty(player.videoElement, 'videoHeight', 720);
    setMediaElementProperty(player.videoElement, 'duration', 120);

    player.playPauseBtn.disabled = true;
    player.seekSlider.disabled = true;

    await player.onVideoLoaded();

    expect(player.canvas.width).toBe(1280);
    expect(player.canvas.height).toBe(720);
    expect(player.resolutionValue.textContent).toBe('1280x720');
    expect(player.isVideoLoaded).toBe(true);
    expect(player.playPauseBtn.disabled).toBe(false);
    expect(player.seekSlider.disabled).toBe(false);
    expect(player.playerStatusEl.textContent).toContain('Video loaded');
    expect(player.renderer.setupPipeline).toHaveBeenCalledWith(player.videoElement, player.dotSize);
  });

  it('reports renderer setup errors for videos', async () => {
    const player = await createTestPlayer();
    player.renderer = { setupPipeline: vi.fn().mockRejectedValue(new Error('boom')) };

    setMediaElementProperty(player.videoElement, 'videoWidth', 640);
    setMediaElementProperty(player.videoElement, 'videoHeight', 480);

    await player.onVideoLoaded();

    expect(player.isVideoLoaded).toBe(false);
    expect(player.playerStatusEl.textContent).toContain('Failed to initialize renderer for video: boom');
  });

  it('updates playback state when the video ends', async () => {
    const player = await createTestPlayer();
    if (!globalThis.cancelAnimationFrame) {
      globalThis.cancelAnimationFrame = () => {};
    }
    const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});

    player.isVideoPlaying = true;
    player.videoAnimationFrame = 42;

    player.onVideoEnded();

    expect(player.isVideoPlaying).toBe(false);
    expect(player.playPauseBtn.textContent).toBe('Play');
    expect(cancelSpy).toHaveBeenCalledWith(42);
    expect(player.playerStatusEl.textContent).toBe('Playback ended');
  });

  it('starts the video render loop when playing', async () => {
    const player = await createTestPlayer();
    if (!globalThis.requestAnimationFrame) {
      globalThis.requestAnimationFrame = () => 0;
    }
    const rafSpy = vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation(() => 99);
    const nowSpy = vi.spyOn(performance, 'now');

    nowSpy.mockReturnValueOnce(1000).mockReturnValueOnce(1016);

    player.isVideoPlaying = true;
    player.activeInputSource = 'video-file';
    player.renderFrame = vi.fn();
    player.updateFPS = vi.fn();

    player.startVideoRenderLoop();

    expect(player.renderFrame).toHaveBeenCalled();
    expect(player.updateFPS).toHaveBeenCalledWith(16);
    expect(rafSpy).toHaveBeenCalled();
    expect(player.videoAnimationFrame).toBe(99);
  });

  it('pauses playback when a render error occurs', async () => {
    const player = await createTestPlayer();
    const pauseSpy = vi.spyOn(player, 'pauseVideo').mockImplementation(() => {});
    const statusSpy = vi.spyOn(player, 'setStatus');

    player.isVideoPlaying = true;
    player.activeInputSource = 'video-file';
    player.renderFrame = vi.fn(() => {
      throw new Error('boom');
    });

    player.startVideoRenderLoop();

    expect(statusSpy).toHaveBeenCalledWith('Render error: boom', player.playerStatusEl);
    expect(pauseSpy).toHaveBeenCalled();
  });

  it('handles video element errors by disabling controls', async () => {
    const player = await createTestPlayer();
    if (!globalThis.cancelAnimationFrame) {
      globalThis.cancelAnimationFrame = () => {};
    }
    const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});

    player.isVideoLoaded = true;
    player.isVideoPlaying = true;
    player.playPauseBtn.disabled = false;
    player.seekSlider.disabled = false;
    player.videoAnimationFrame = 7;

    player.onVideoError({ target: { error: { code: 4 } } });

    expect(cancelSpy).toHaveBeenCalledWith(7);
    expect(player.isVideoLoaded).toBe(false);
    expect(player.isVideoPlaying).toBe(false);
    expect(player.playPauseBtn.disabled).toBe(true);
    expect(player.seekSlider.disabled).toBe(true);
    expect(player.playPauseBtn.textContent).toBe('Play');
    expect(player.playerStatusEl.textContent).toContain('Video error:');
  });
});

describe('CyberVision - Camera Render Loop', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('stops the camera when a render error occurs', async () => {
    const player = await createTestPlayer();
    const stopSpy = vi.spyOn(player, 'stopCamera').mockImplementation(() => {});
    const statusSpy = vi.spyOn(player, 'setStatus');

    player.isCameraRunning = true;
    player.activeInputSource = 'camera';
    player.renderFrame = vi.fn(() => {
      throw new Error('boom');
    });

    player.startCameraRenderLoop();

    expect(statusSpy).toHaveBeenCalledWith('Render Error: boom', player.statusEl);
    expect(stopSpy).toHaveBeenCalled();
  });
});

describe('CyberVision - Renderer Switching', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('cleans up old renderer and replaces the canvas for WebGL', async () => {
    const player = await createTestPlayer();
    const oldRenderer = { cleanup: vi.fn() };
    const oldCanvas = player.canvas;

    oldCanvas.width = 320;
    oldCanvas.height = 240;
    player.renderer = oldRenderer;

    await player.switchToRenderer('webgl');

    expect(oldRenderer.cleanup).toHaveBeenCalled();
    expect(player.rendererType).toBe('webgl');
    expect(player.gpuStatus.textContent).toBe('WebGL');
    expect(player.gpuStatus.style.color).toBe('#60a5fa');
    expect(player.canvas).not.toBe(oldCanvas);
    expect(player.canvas.width).toBe(320);
    expect(player.canvas.height).toBe(240);
  });

  it('sets WebGPU status when switching renderer', async () => {
    const player = await createTestPlayer();

    await player.switchToRenderer('webgpu');

    expect(player.rendererType).toBe('webgpu');
    expect(player.gpuStatus.textContent).toContain('WebGPU');
    expect(player.gpuStatus.style.color).toBe('#34d399');
  });
});

describe('CyberVision - Renderer Toggle', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('forces WebGL when WebGPU is unavailable', async () => {
    const player = await createTestPlayer();
    const switchSpy = vi.spyOn(player, 'switchToRenderer').mockImplementation(() => {});

    player.webgpuAvailable = false;
    player.webglToggle.checked = false;

    player.handleRendererToggle(false);

    expect(player.webglToggle.checked).toBe(true);
    expect(switchSpy).not.toHaveBeenCalled();
  });

  it('restarts the camera when toggling renderer mid-stream', async () => {
    const player = await createTestPlayer();
    const stopSpy = vi.spyOn(player, 'stopCamera').mockImplementation(() => {});
    const startSpy = vi.spyOn(player, 'startCamera').mockImplementation(() => {});

    vi.spyOn(player, 'switchToRenderer').mockImplementation((target) => {
      player.rendererType = target;
    });

    player.activeInputSource = 'camera';
    player.isCameraRunning = true;
    player.webgpuAvailable = true;

    player.handleRendererToggle(true);

    expect(stopSpy).toHaveBeenCalled();
    expect(startSpy).toHaveBeenCalled();
    expect(player.statusEl.textContent).toContain('Switched to WEBGL.');
  });

  it('restarts video playback after a renderer switch', async () => {
    const player = await createTestPlayer();
    const pauseSpy = vi.spyOn(player, 'pauseVideo').mockImplementation(() => {});
    const playSpy = vi.spyOn(player, 'playVideo').mockImplementation(() => {});

    vi.spyOn(player, 'switchToRenderer').mockImplementation((target) => {
      player.rendererType = target;
    });

    player.activeInputSource = 'video-file';
    player.isVideoPlaying = true;
    player.isVideoLoaded = true;
    player.webgpuAvailable = true;

    player.handleRendererToggle(true);

    expect(pauseSpy).toHaveBeenCalled();
    expect(playSpy).toHaveBeenCalled();
  });
});

describe('CyberVision - Segmentation', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('loads the segmentation model and updates progress UI', async () => {
    const player = await createTestPlayer();
    const { PortraitSegmentation } = await import('/lib/ml-inference.js');

    const loadSpy = vi
      .spyOn(PortraitSegmentation.prototype, 'loadModel')
      .mockImplementation(async (_path, onProgress) => {
        onProgress({ stage: 'downloading', progress: 0.5 });
        onProgress({ stage: 'initializing' });
        onProgress({ stage: 'ready' });
      });

    await player.loadSegmentationModel();

    expect(player.segmentationModelLoaded).toBe(true);
    expect(player.segmentationML).toBeInstanceOf(PortraitSegmentation);
    expect(player.segmentationLoadingText.textContent).toBe('Model ready!');
    expect(player.segmentationLoading.style.display).toBe('none');
    expect(loadSpy).toHaveBeenCalledWith(
      '/static/models/segmentation.onnx',
      expect.any(Function)
    );
  });

  it('shows errors when the segmentation model fails to load', async () => {
    const player = await createTestPlayer();
    const { PortraitSegmentation } = await import('/lib/ml-inference.js');

    vi.spyOn(PortraitSegmentation.prototype, 'loadModel').mockRejectedValue(new Error('boom'));

    await player.loadSegmentationModel();

    expect(player.segmentationModelLoaded).toBe(false);
    expect(player.segmentationLoadingText.textContent).toBe('Error: Failed to load model.');
    expect(player.segmentationLoading.style.display).toBe('flex');
  });

  it('updates background preview and renderer on upload', async () => {
    const player = await createTestPlayer();
    const originalFileReader = globalThis.FileReader;
    const originalImage = globalThis.Image;

    class MockFileReader {
      readAsDataURL() {
        if (this.onload) {
          this.onload({ target: { result: 'data:image/png;base64,abc' } });
        }
      }
    }

    class MockImage {
      constructor() {
        this._src = '';
        this.onload = null;
      }
      set src(value) {
        this._src = value;
        if (this.onload) {
          this.onload();
        }
      }
      get src() {
        return this._src;
      }
    }

    globalThis.FileReader = MockFileReader;
    globalThis.Image = MockImage;

    player.renderer = { updateBackgroundImage: vi.fn() };
    player.setupEventListeners();

    const file = new File(['data'], 'bg.png', { type: 'image/png' });
    Object.defineProperty(player.segmentationBackgroundUpload, 'files', {
      value: [file],
      configurable: true,
    });

    try {
      player.segmentationBackgroundUpload.dispatchEvent(new Event('change'));

      expect(player.segmentationBackgroundImage).toBeInstanceOf(MockImage);
      expect(player.backgroundPreviewImg.src).toBe('data:image/png;base64,abc');
      expect(player.backgroundPreview.style.display).toBe('block');
      expect(player.renderer.updateBackgroundImage).toHaveBeenCalledWith(
        player.segmentationBackgroundImage
      );
    } finally {
      globalThis.FileReader = originalFileReader;
      globalThis.Image = originalImage;
    }
  });

  it('falls back to passthrough when segmentation is unavailable', async () => {
    const player = await createTestPlayer();
    player.rendererType = 'webgpu';
    player.renderer = { renderPassthrough: vi.fn() };

    await player.renderSegmentation({});

    expect(player.renderer.renderPassthrough).toHaveBeenCalled();
  });

  it('runs segmentation inference and renders the mask', async () => {
    const player = await createTestPlayer();
    const sourceVideo = {};
    const maskData = { data: [1] };
    const processedMask = { data: [2] };

    player.rendererType = 'webgpu';
    player.segmentationModelLoaded = true;
    player.segmentationFrameSkip = 1;
    player.segmentationFrameCounter = 0;
    player.segmentationMask = null;
    player.segmentationML = {
      segmentFrame: vi.fn().mockResolvedValue(maskData),
      postprocessMask: vi.fn().mockReturnValue(processedMask),
    };
    player.renderer = {
      renderSegmentation: vi.fn(),
      renderPassthrough: vi.fn(),
    };

    await player.renderSegmentation(sourceVideo);

    expect(player.segmentationML.segmentFrame).toHaveBeenCalledWith(sourceVideo);
    expect(player.segmentationML.postprocessMask).toHaveBeenCalledWith(maskData);
    expect(player.renderer.renderSegmentation).toHaveBeenCalledWith(
      sourceVideo,
      player.segmentationMode_state,
      player.segmentationBlurRadius_state,
      processedMask,
      player.segmentationSoftEdges_state,
      player.segmentationGlow_state
    );
  });
});

describe('CyberVision - Screenshot Capture', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('writes status after saving a screenshot', async () => {
    const player = await createTestPlayer();
    const blob = new Blob(['data'], { type: 'image/png' });

    player.isRunning = true;
    player.canvas.toBlob = vi.fn((callback) => callback(blob));

    if (!URL.createObjectURL) {
      URL.createObjectURL = () => 'blob:stub';
    }
    if (!URL.revokeObjectURL) {
      URL.revokeObjectURL = () => {};
    }

    const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    player.takeScreenshot();

    expect(player.statusEl.textContent).toContain('Screenshot saved: cybervision-screenshot-');
    expect(createSpy).toHaveBeenCalledWith(blob);
    expect(revokeSpy).toHaveBeenCalledWith('blob:mock');
    expect(clickSpy).toHaveBeenCalled();
  });

  it('reports errors when screenshot creation fails', async () => {
    const player = await createTestPlayer();

    player.isRunning = true;
    player.canvas.toBlob = vi.fn((callback) => callback(null));

    player.takeScreenshot();

    expect(player.statusEl.textContent).toBe('Screenshot error: Failed to create image blob.');
  });
});
