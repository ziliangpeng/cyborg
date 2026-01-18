import { test, expect } from '@playwright/test';
import { switchToEffectTab } from './test-helpers.js';

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

    // Wait for initialization to complete (gpuStatus changes from "Checking..." to "WebGL")
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Check no "not supported" error
    const statusText = await page.locator('#status').textContent();
    expect(statusText).not.toContain('Not supported');
    expect(statusText).not.toContain('Neither WebGPU nor WebGL2 is available');

    // Check no critical errors
    const criticalErrors = consoleErrors.filter(e =>
      e.includes('not supported') ||
      e.includes('Failed to initialize')
    );
    expect(criticalErrors).toHaveLength(0);
  });

  test('should have start button enabled', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

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
      await switchToEffectTab(page, effect);
      const radio = page.locator(`input[value="${effect}"]`);
      // Use evaluate() instead of check({ force: true }) because Playwright's
      // check() still attempts to scroll into view even with force: true,
      // which fails for elements in inactive tabs with display: none
      await radio.evaluate(el => el.click());
      await expect(radio).toBeChecked();
    }
  });

  test('should show/hide effect controls when switching effects', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Select halftone effect
    await switchToEffectTab(page, 'halftone');
    await page.locator('input[value="halftone"]').check({ force: true });
    await expect(page.locator('#halftoneControls')).toBeVisible();
    await expect(page.locator('#clusteringControls')).not.toBeVisible();

    // Select clustering effect
    await switchToEffectTab(page, 'clustering');
    await page.locator('input[value="clustering"]').check({ force: true });
    await expect(page.locator('#clusteringControls')).toBeVisible();
    await expect(page.locator('#halftoneControls')).not.toBeVisible();

    // Select edges effect
    await switchToEffectTab(page, 'edges');
    await page.locator('input[value="edges"]').check({ force: true });
    await expect(page.locator('#edgesControls')).toBeVisible();
    await expect(page.locator('#clusteringControls')).not.toBeVisible();

    // Select original (no controls should be visible)
    await switchToEffectTab(page, 'original');
    await page.locator('input[value="original"]').check({ force: true });
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
      await switchToEffectTab(page, effect);
      // Use evaluate() instead of check({ force: true }) because Playwright's
      // check() still attempts to scroll into view even with force: true,
      // which fails for elements in inactive tabs with display: none
      await page.locator(`input[value="${effect}"]`).evaluate(el => el.click());
      // No need to wait - effect switching is synchronous in WebGL
    }

    // Check no errors occurred during effect switching
    expect(consoleErrors.filter(e => !e.includes('DevTools') && !e.includes('favicon')).length).toBe(0);
  });

  test('should have correct initial UI state', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Check initial effect is selected
    const originalRadio = page.locator('input[value="original"]');
    await expect(originalRadio).toBeChecked();

    // Check initial button states
    await expect(page.locator('#startBtn')).toBeEnabled();
    await expect(page.locator('#stopBtn')).toBeDisabled();

    // Check initial stats (some may show "0" or "0 ms" instead of "-" after init)
    const fpsValue = await page.locator('#fpsValue').textContent();
    expect(['-', '0']).toContain(fpsValue);

    const latencyValue = await page.locator('#latencyValue').textContent();
    expect(['-', '0 ms']).toContain(latencyValue);

    const resolutionValue = await page.locator('#resolutionValue').textContent();
    expect(resolutionValue).toBe('-');
  });

  test('should update slider values when interacting with controls', async ({ page }) => {
    await page.goto('/?force-webgl=true');

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Select halftone effect
    await switchToEffectTab(page, 'halftone');
    await page.locator('input[value="halftone"]').check({ force: true });

    // Wait for halftone controls to be visible
    await expect(page.locator('#halftoneControls')).toBeVisible();

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

    // Wait for initialization to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 5000 });

    // Select mosaic effect
    await switchToEffectTab(page, 'mosaic');
    await page.locator('input[value="mosaic"]').check({ force: true });

    // Wait for mosaic controls to be visible
    await expect(page.locator('#mosaicControls')).toBeVisible();

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
