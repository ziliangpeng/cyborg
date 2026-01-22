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

    <!-- Tab structure -->
    <div class="tab-button" data-tab="effects">Effects</div>
    <div class="tab-button" data-tab="settings">Settings</div>
    <div class="tab-content" id="tab-effects"></div>
    <div class="tab-content" id="tab-settings"></div>

    <!-- Effect buttons -->
    <button class="effect-btn" data-effect="original">Original</button>
    <button class="effect-btn" data-effect="halftone">Halftone</button>
    <button class="effect-btn" data-effect="clustering">Clustering</button>
    <button class="effect-btn" data-effect="edges">Edges</button>
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
