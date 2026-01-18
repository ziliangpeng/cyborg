import { test, expect } from '@playwright/test';
import { switchToEffectTab } from './test-helpers.js';

test.describe('CyberVision E2E - WebGPU Path (macOS only)', () => {
  test.beforeEach(async ({ page }) => {
    // Check if WebGPU is available
    const webgpuAvailable = await page.evaluate(async () => {
      if (!('gpu' in navigator)) return false;
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
      } catch {
        return false;
      }
    });

    test.skip(!webgpuAvailable, 'WebGPU not available in this environment');
  });

  test('should load page with WebGPU renderer', async ({ page }) => {
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

    // Wait for initialization to complete (gpuStatus changes from "Checking..." to "WebGPU")
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Check page loaded successfully
    await expect(page.locator('h1')).toContainText('CyberVision');

    // Check no "not supported" error
    const statusText = await page.locator('#status').textContent();
    expect(statusText).not.toContain('Not supported');
    expect(statusText).not.toContain('Neither WebGPU nor WebGL2 is available');

    // Check WebGPU is active
    const gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toContain('WebGPU');

    // Check no critical errors
    const criticalErrors = consoleErrors.filter(e =>
      e.includes('not supported') ||
      e.includes('Failed to initialize')
    );
    expect(criticalErrors).toHaveLength(0);
  });

  test('should have pixelsort effect available (WebGPU only)', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Pixelsort is only available on WebGPU
    await switchToEffectTab(page, 'pixelsort');
    const pixelsortButton = page.locator('button[data-effect="pixelsort"]');
    await expect(pixelsortButton).toBeVisible();

    await pixelsortButton.click();
    await expect(pixelsortButton).toHaveClass(/selected/);

    // Controls should be visible
    await expect(page.locator('#pixelSortControls')).toBeVisible();
  });

  test('should have kaleidoscope effect available', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    await switchToEffectTab(page, 'kaleidoscope');
    const kaleidoscopeButton = page.locator('button[data-effect="kaleidoscope"]');
    await expect(kaleidoscopeButton).toBeVisible();

    await kaleidoscopeButton.click();
    await expect(kaleidoscopeButton).toHaveClass(/selected/);

    // Controls should be visible
    await expect(page.locator('#kaleidoscopeControls')).toBeVisible();
  });

  test('should be able to select all WebGPU effects', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    const effects = [
      'original',
      'halftone',
      'clustering',
      'edges',
      'mosaic',
      'chromatic',
      'glitch',
      'thermal',
      'pixelsort',
      'kaleidoscope'
    ];

    for (const effect of effects) {
      await switchToEffectTab(page, effect);
      const button = page.locator(`button[data-effect="${effect}"]`);
      await button.click({ force: true });
      await expect(button).toHaveClass(/selected/);
    }
  });

  test('should switch between WebGPU and WebGL renderers', async ({ page }) => {
    await page.goto('/');

    // Wait for initial WebGPU initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Initially should be WebGPU
    let gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toContain('WebGPU');

    // Toggle to WebGL
    const webglToggle = page.locator('#webglToggle');
    await webglToggle.click();

    // Wait for renderer switch to complete
    await expect(page.locator('#gpuStatus')).toHaveText('WebGL', { timeout: 3000 });

    // Should now be WebGL
    gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toBe('WebGL');

    // Toggle back to WebGPU
    await webglToggle.uncheck();

    // Wait for renderer switch back to WebGPU
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 3000 });

    // Should be WebGPU again
    gpuStatus = await page.locator('#gpuStatus').textContent();
    expect(gpuStatus).toContain('WebGPU');
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

    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    const effects = [
      'original',
      'halftone',
      'clustering',
      'edges',
      'mosaic',
      'chromatic',
      'glitch',
      'thermal',
      'pixelsort',
      'kaleidoscope'
    ];

    for (const effect of effects) {
      await switchToEffectTab(page, effect);
      await page.locator(`button[data-effect="${effect}"]`).click();
      // Small delay to allow effect to be applied
      await page.waitForTimeout(100);
    }

    // Check no errors occurred during effect switching
    const relevantErrors = consoleErrors.filter(e =>
      !e.includes('DevTools') && !e.includes('favicon')
    );
    expect(relevantErrors).toHaveLength(0);
  });

  test('should show pixel sort iterations control only for bubble sort', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Select pixelsort effect
    await switchToEffectTab(page, 'pixelsort');
    await page.locator('button[data-effect="pixelsort"]').click();

    const iterationsGroup = page.locator('#pixelSortIterationsGroup');
    const algorithmSelect = page.locator('#pixelSortAlgorithm');

    // Select bitonic (default) - iterations should be hidden
    await algorithmSelect.selectOption('bitonic');
    await expect(iterationsGroup).not.toBeVisible();

    // Select bubble - iterations should be visible
    await algorithmSelect.selectOption('bubble');
    await expect(iterationsGroup).toBeVisible();

    // Select quicksort - iterations should be hidden
    await algorithmSelect.selectOption('quicksort');
    await expect(iterationsGroup).not.toBeVisible();
  });

  test('should update kaleidoscope controls', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Select kaleidoscope effect
    await switchToEffectTab(page, 'kaleidoscope');
    await page.locator('button[data-effect="kaleidoscope"]').click();

    // Update segments slider
    const segmentsSlider = page.locator('#segmentsSlider');
    const segmentsValue = page.locator('#segmentsValue');

    await segmentsSlider.fill('12');
    await expect(segmentsValue).toHaveText('12');

    // Update rotation speed slider
    const rotationSpeedSlider = page.locator('#rotationSpeedSlider');
    const rotationSpeedValue = page.locator('#rotationSpeedValue');

    await rotationSpeedSlider.fill('0.5');
    const speedText = await rotationSpeedValue.textContent();
    expect(parseFloat(speedText)).toBeCloseTo(0.5, 1);
  });

  test('should not show mosaic info when using dominant mode with WebGPU', async ({ page }) => {
    await page.goto('/');

    // Wait for initialization
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Select mosaic effect
    await switchToEffectTab(page, 'mosaic');
    await page.locator('button[data-effect="mosaic"]').click();

    // Switch to dominant mode
    await page.locator('#mosaicMode').selectOption('dominant');

    // Info should NOT be visible with WebGPU (only shows with WebGL fallback)
    await expect(page.locator('#mosaicInfo')).not.toBeVisible();
  });
});
