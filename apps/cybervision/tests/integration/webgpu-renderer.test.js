import { test, expect } from '@playwright/test';

test.describe('WebGPU Renderer Integration (macOS only)', () => {
  test.beforeEach(async ({ page }) => {
    // Check if WebGPU is available
    const webgpuAvailable = await page.evaluate(() => {
      return 'gpu' in navigator;
    });

    test.skip(!webgpuAvailable, 'WebGPU not available in this environment');
  });

  test('should initialize WebGPU context', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Check that page loads without "not supported" error
    const statusText = await page.locator('#status').textContent();
    expect(statusText).not.toContain('Not supported');
    expect(statusText).not.toContain('Error');

    // Check GPU status shows WebGPU
    const gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toContain('WebGPU');
  });

  test('should compile all WGSL shaders without errors', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    page.on('pageerror', error => {
      consoleErrors.push(error.message);
    });

    await page.goto('/');

    // Wait for initialization to complete (ensures shaders are compiled)
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Check for shader compilation errors
    const shaderErrors = consoleErrors.filter(err =>
      err.toLowerCase().includes('shader') ||
      err.toLowerCase().includes('wgsl') ||
      err.toLowerCase().includes('pipeline')
    );

    expect(shaderErrors, `Shader errors found: ${shaderErrors.join(', ')}`).toHaveLength(0);
  });

  test('should create WebGPU pipelines successfully', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');

    // Wait for initialization to complete (ensures pipelines are created)
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Check for pipeline creation errors
    const pipelineErrors = consoleErrors.filter(err =>
      err.includes('pipeline') || err.includes('createRenderPipeline')
    );

    expect(pipelineErrors, `Pipeline errors found: ${pipelineErrors.join(', ')}`).toHaveLength(0);
  });

  test('should have all WebGPU effect render methods', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Wait for app to be available
    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    // Check that renderer has required methods
    const hasRenderMethods = await page.evaluate(() => {
      const app = window.cyberVisionApp;
      if (!app || !app.renderer) {
        return { success: false, error: 'App or renderer not initialized' };
      }

      const requiredMethods = [
        'renderHalftone',
        'renderClustering',
        'renderEdges',
        'renderMosaic',
        'renderAscii',
        'renderOilPaint',
        'renderChromatic',
        'renderGlitch',
        'renderThermal',
        'renderPixelSort',
        'renderKaleidoscope',
        'renderPassthrough',
        'renderSegmentation'
      ];

      const missingMethods = requiredMethods.filter(
        method => typeof app.renderer[method] !== 'function'
      );

      return {
        success: missingMethods.length === 0,
        missing: missingMethods
      };
    });

    expect(hasRenderMethods.success, `Missing methods: ${hasRenderMethods.missing?.join(', ')}`).toBe(true);
  });

  test('should render effects without throwing', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    page.on('pageerror', error => {
      consoleErrors.push(error.message);
    });

    await page.goto('/');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Wait for app to be available
    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    // Try rendering with a mock video source
    const renderResult = await page.evaluate(async () => {
      const app = window.cyberVisionApp;
      if (!app || !app.renderer) {
        return { success: false, error: 'App not initialized' };
      }

      // Create a mock video element
      const mockVideo = document.createElement('canvas');
      mockVideo.width = 640;
      mockVideo.height = 480;
      const ctx = mockVideo.getContext('2d');
      ctx.fillStyle = 'blue';
      ctx.fillRect(0, 0, 640, 480);

      try {
        // Setup pipeline first for WebGPU and await its completion
        await app.renderer.setupPipeline(mockVideo, 8);

        // Test each render method with mock video
        app.renderer.renderPassthrough(mockVideo);
        app.renderer.renderHalftone(mockVideo, false);
        app.renderer.renderClustering(mockVideo, 'quantization-kmeans', 8, 0.1);
        app.renderer.renderEdges(mockVideo, 'sobel', 0.1, false, false, [1, 1, 1], 1);
        app.renderer.renderMosaic(mockVideo, 8, 'center');
        app.renderer.renderAscii(mockVideo, 10, true, true);
        app.renderer.renderOilPaint(mockVideo, 4, 6);
        app.renderer.renderChromatic(mockVideo, 10, 'radial', 0.5, 0.5);
        app.renderer.renderGlitch(mockVideo, 'slices', 12, 24, 4, 0.15, 0.3);
        app.renderer.renderThermal(mockVideo, 'classic', 1.0, false);
        app.renderer.renderPixelSort(mockVideo, 'preset', 'horizontal', 0, 0.25, 0.75, 'brightness', 'luminance', 'ascending', 'bitonic', 50);
        app.renderer.renderKaleidoscope(mockVideo, 8, 0.0);

        // Test segmentation with mock mask data
        const maskData = new Uint8Array(256 * 256).fill(128);
        app.renderer.renderSegmentation(mockVideo, 'blur', 10, maskData, false, false);

        return { success: true };
      } catch (err) {
        return { success: false, error: err.message, stack: err.stack };
      }
    });

    expect(renderResult.success, `Render error: ${renderResult.error}`).toBe(true);
    expect(consoleErrors.filter(e => e.includes('render')).length).toBe(0);
  });
});
