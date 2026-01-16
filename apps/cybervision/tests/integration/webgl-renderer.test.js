import { test, expect } from '@playwright/test';

test.describe('WebGL Renderer Integration', () => {
  test('should initialize WebGL2 context', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Check that page loads without "not supported" error
    const statusText = await page.locator('#status').textContent();
    expect(statusText).not.toContain('Not supported');
    expect(statusText).not.toContain('Error');
  });

  test('should compile all WebGL shaders without errors', async ({ page }) => {
    const consoleErrors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/?force-webgl=true');

    // Wait for initialization
    await page.waitForTimeout(1000);

    // Check for shader compilation errors
    const shaderErrors = consoleErrors.filter(err =>
      err.includes('shader') ||
      err.includes('GLSL') ||
      err.includes('compile')
    );

    expect(shaderErrors, `Shader errors found: ${shaderErrors.join(', ')}`).toHaveLength(0);
  });

  test('should load all shader files without 404s', async ({ page }) => {
    const failedRequests = [];
    page.on('requestfailed', request => {
      failedRequests.push(request.url());
    });

    await page.goto('/?force-webgl=true');

    // Wait for all shader files to load
    await page.waitForTimeout(2000);

    // Check for 404s on shader files
    const shaderNotFound = failedRequests.filter(url =>
      url.includes('.glsl') || url.includes('.wgsl')
    );

    expect(shaderNotFound, `Shader files not found: ${shaderNotFound.join(', ')}`).toHaveLength(0);
  });

  test('should have all WebGL effect render methods', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for app to initialize
    await page.waitForFunction(() => window.cyberVisionApp != null, { timeout: 5000 });

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
        'renderChromatic',
        'renderGlitch',
        'renderThermal',
        'renderPassthrough'
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

    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Wait for app to be available
    await page.waitForFunction(() => window.cyberVisionApp != null, { timeout: 5000 });

    // Try rendering with a mock video source
    const renderResult = await page.evaluate(() => {
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
        // Test each render method with mock video
        app.renderer.renderPassthrough(mockVideo);
        app.renderer.renderHalftone(mockVideo, 8, false);
        app.renderer.renderClustering(mockVideo, 'quantization-kmeans', 8, 0.1);
        app.renderer.renderEdges(mockVideo, 'sobel', 0.1, false, false, [1, 1, 1], 1);
        app.renderer.renderMosaic(mockVideo, 8, 'center');
        app.renderer.renderChromatic(mockVideo, 10, 'radial', 0.5, 0.5);
        app.renderer.renderGlitch(mockVideo, 'slices', 12, 24, 4, 0.15, 0.3);
        app.renderer.renderThermal(mockVideo, 'classic', 1.0, false);

        return { success: true };
      } catch (err) {
        return { success: false, error: err.message };
      }
    });

    expect(renderResult.success, `Render error: ${renderResult.error}`).toBe(true);
    expect(consoleErrors.filter(e => e.includes('render')).length).toBe(0);
  });
});
