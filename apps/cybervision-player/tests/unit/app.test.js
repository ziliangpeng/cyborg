import { describe, it, expect, beforeEach } from 'vitest';

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
    <canvas id="video-canvas"></canvas>
    <button id="play-pause-btn">Play</button>
    <input id="seek-slider" type="range" min="0" max="100" value="0" />
    <div id="time-display">0:00 / 0:00</div>
    <div id="status-message"></div>

    <!-- Core UI elements -->
    <div id="status"></div>
    <div id="fpsValue">0</div>
    <div id="latencyValue">0</div>
    <div id="gpuStatus">Unknown</div>
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
      </select>
      <input id="pixelSortThresholdLow" type="range" min="0" max="1" step="0.01" value="0.25" />
      <span id="pixelSortThresholdLowValue">0.25</span>
      <input id="pixelSortThresholdHigh" type="range" min="0" max="1" step="0.01" value="0.75" />
      <span id="pixelSortThresholdHighValue">0.75</span>
      <select id="pixelSortKey">
        <option value="luminance">Luminance</option>
      </select>
      <select id="pixelSortOrder">
        <option value="ascending">Ascending</option>
      </select>
      <select id="pixelSortAlgorithm">
        <option value="bitonic">Bitonic</option>
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

describe('VideoPlayer - Time Formatting', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should format time correctly for valid seconds', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.formatTime(0)).toBe('0:00');
    expect(player.formatTime(30)).toBe('0:30');
    expect(player.formatTime(60)).toBe('1:00');
    expect(player.formatTime(90)).toBe('1:30');
    expect(player.formatTime(125)).toBe('2:05');
    expect(player.formatTime(3661)).toBe('61:01');
  });

  it('should handle NaN input', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.formatTime(NaN)).toBe('0:00');
  });

  it('should pad seconds with leading zero', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.formatTime(5)).toBe('0:05');
    expect(player.formatTime(65)).toBe('1:05');
  });
});

describe('VideoPlayer - Effect Parameters', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should initialize with default effect parameters', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.dotSize).toBe(8);
    expect(player.useRandomColors).toBe(false);
    expect(player.colorCount).toBe(8);
    expect(player.segments).toBe(8);
    expect(player.rotationSpeed).toBe(0.0);
    expect(player.currentEffect).toBe('original');
  });

  it('should initialize video player state', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.isPlaying).toBe(false);
    expect(player.isVideoLoaded).toBe(false);
    expect(player.renderer).toBe(null);
    expect(player.animationFrame).toBe(null);
  });

  it('should initialize segmentation state variables correctly', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    expect(player.segmentationML).toBe(null);
    expect(player.segmentationModelLoaded).toBe(false);
    expect(player.segmentationModeValue).toBe('blackout');
    expect(player.segmentationBlurRadiusValue_state).toBe(10);
  });
});

describe('VideoPlayer - Local Video Loading', () => {
  beforeEach(() => {
    setupMockDOM();
  });

  it('should load local video file and create blob URL', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    const mockFile = new File(['video content'], 'test.mp4', { type: 'video/mp4' });
    player.loadLocalVideo(mockFile);

    expect(player.videoElement.src).toContain('blob:');
    expect(document.getElementById('currentFile').textContent).toBe('test.mp4');
  });

  it('should revoke previous blob URL when loading new video', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL');

    const file1 = new File(['video1'], 'video1.mp4', { type: 'video/mp4' });
    const file2 = new File(['video2'], 'video2.mp4', { type: 'video/mp4' });

    player.loadLocalVideo(file1);
    const firstBlobUrl = player.currentBlobUrl;

    player.loadLocalVideo(file2);

    expect(revokeObjectURL).toHaveBeenCalledWith(firstBlobUrl);
  });

  it('should update current file display with filename', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    const mockFile = new File(['content'], 'my-video.mov', { type: 'video/quicktime' });
    player.loadLocalVideo(mockFile);

    expect(player.currentFileDisplay.textContent).toBe('my-video.mov');
  });

  it('should show loading status when loading video', async () => {
    const { VideoPlayer } = await import('../../webroot/static/app.js');
    const player = new VideoPlayer();

    const mockFile = new File(['content'], 'test.mp4', { type: 'video/mp4' });
    player.loadLocalVideo(mockFile);

    const status = document.getElementById('status');
    expect(status.textContent).toContain('Loading video');
  });
});
