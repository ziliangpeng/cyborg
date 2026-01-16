import { test, expect } from '@playwright/test';

test.describe('CyberVision E2E - WebGL Path', () => {
  test('should load page with WebGL fallback', async ({ page }) => {
    // Monitor console for errors
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

    // Check page loaded successfully
    await expect(page.locator('h1')).toContainText('CyberVision');

    // Check no "not supported" error
    const statusText = await page.locator('#status').textContent();
    expect(statusText).not.toContain('Not supported');
    expect(statusText).not.toContain('Neither WebGPU nor WebGL2 is available');

    // Check WebGL is active
    const gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toBe('WebGL');

    // Check no critical errors
    const criticalErrors = consoleErrors.filter(e =>
      e.includes('not supported') ||
      e.includes('Failed to initialize')
    );
    expect(criticalErrors).toHaveLength(0);
  });

  test('should have start button enabled', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization
    await page.waitForTimeout(1000);

    // Check start button is enabled
    const startBtn = page.locator('#startBtn');
    await expect(startBtn).toBeEnabled();
  });

  test('should be able to select all effect radio buttons', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    const effects = [
      'original',
      'halftone',
      'clustering',
      'edges',
      'mosaic',
      'chromatic',
      'glitch',
      'thermal'
    ];

    for (const effect of effects) {
      const radio = page.locator(`input[value="${effect}"]`);
      await radio.check();
      await expect(radio).toBeChecked();
    }
  });

  test('should show/hide effect controls when switching effects', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Select halftone effect
    await page.locator('input[value="halftone"]').check();
    await expect(page.locator('#halftoneControls')).toBeVisible();
    await expect(page.locator('#clusteringControls')).not.toBeVisible();

    // Select clustering effect
    await page.locator('input[value="clustering"]').check();
    await expect(page.locator('#clusteringControls')).toBeVisible();
    await expect(page.locator('#halftoneControls')).not.toBeVisible();

    // Select edges effect
    await page.locator('input[value="edges"]').check();
    await expect(page.locator('#edgesControls')).toBeVisible();
    await expect(page.locator('#clusteringControls')).not.toBeVisible();

    // Select original (no controls should be visible)
    await page.locator('input[value="original"]').check();
    await expect(page.locator('#halftoneControls')).not.toBeVisible();
    await expect(page.locator('#clusteringControls')).not.toBeVisible();
    await expect(page.locator('#edgesControls')).not.toBeVisible();
  });

  test('should not have console errors when switching effects', async ({ page }) => {
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

    const effects = ['original', 'halftone', 'clustering', 'edges', 'mosaic', 'chromatic', 'glitch', 'thermal'];

    for (const effect of effects) {
      await page.locator(`input[value="${effect}"]`).check();
      await page.waitForTimeout(100); // Small delay to allow any async errors
    }

    // Check no errors occurred during effect switching
    expect(consoleErrors.filter(e => !e.includes('DevTools')).length).toBe(0);
  });

  test('should have correct initial UI state', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Check initial effect is selected
    const originalRadio = page.locator('input[value="original"]');
    await expect(originalRadio).toBeChecked();

    // Check initial button states
    await expect(page.locator('#startBtn')).toBeEnabled();
    await expect(page.locator('#stopBtn')).toBeDisabled();

    // Check initial stats
    const fpsValue = await page.locator('#fpsValue').textContent();
    expect(fpsValue).toBe('-');

    const latencyValue = await page.locator('#latencyValue').textContent();
    expect(latencyValue).toBe('-');

    const resolutionValue = await page.locator('#resolutionValue').textContent();
    expect(resolutionValue).toBe('-');
  });

  test('should update slider values when interacting with controls', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Select halftone effect
    await page.locator('input[value="halftone"]').check();

    // Check initial dot size value
    const dotSizeValue = page.locator('#dotSizeValue');
    const initialValue = await dotSizeValue.textContent();

    // Change dot size slider
    const dotSizeSlider = page.locator('#dotSizeSlider');
    await dotSizeSlider.fill('16');

    // Check value updated
    await expect(dotSizeValue).toHaveText('16');
    expect(await dotSizeValue.textContent()).not.toBe(initialValue);
  });

  test('should handle renderer toggle', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Check WebGL toggle is checked (since we forced WebGL)
    const webglToggle = page.locator('#webglToggle');
    await expect(webglToggle).toBeChecked();

    // Status should indicate WebGL is active
    const gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toBe('WebGL');
  });

  test('should display mosaic info when using dominant mode with WebGL', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Select mosaic effect
    await page.locator('input[value="mosaic"]').check();

    // Initially should be hidden (default is center mode)
    await expect(page.locator('#mosaicInfo')).not.toBeVisible();

    // Switch to dominant mode
    await page.locator('#mosaicMode').selectOption('dominant');

    // Info should now be visible (WebGL falls back to centerSample)
    await expect(page.locator('#mosaicInfo')).toBeVisible();

    // Switch back to center
    await page.locator('#mosaicMode').selectOption('center');

    // Info should be hidden again
    await expect(page.locator('#mosaicInfo')).not.toBeVisible();
  });
});
