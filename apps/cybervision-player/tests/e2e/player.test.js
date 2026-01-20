import { test, expect } from '@playwright/test';

test.describe('CyberVision Video Player', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the page successfully', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('CyberVision Video Player');
    await expect(page.locator('#video-path')).toBeVisible();
    await expect(page.locator('#load-video-btn')).toBeVisible();
    await expect(page.locator('#effect-select')).toBeVisible();
  });

  test('should have all effect options available', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');
    await expect(effectSelect).toBeVisible();

    const options = await effectSelect.locator('option').allTextContents();
    expect(options).toContain('Original');
    expect(options).toContain('Halftone');
    expect(options).toContain('Clustering');
    expect(options).toContain('Kaleidoscope');
  });

  test('should show error message for empty video path', async ({ page }) => {
    // Clear the default video path first
    await page.locator('#video-path').clear();
    await page.click('#load-video-btn');
    await expect(page.locator('#status-message')).toContainText('Please enter a video file path');
  });

  test('should change effect selection', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');

    // Select halftone effect
    await effectSelect.selectOption('halftone');
    await expect(effectSelect).toHaveValue('halftone');

    // Verify effect params UI appears
    const effectParams = page.locator('#effect-params');
    await expect(effectParams).toBeVisible();
  });

  test('should show halftone controls when halftone effect is selected', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');
    await effectSelect.selectOption('halftone');

    // Check for dot size slider
    const dotSizeSlider = page.locator('#dot-size-slider');
    await expect(dotSizeSlider).toBeVisible();

    // Check for random colors checkbox
    const randomColorsCheckbox = page.locator('#random-colors');
    await expect(randomColorsCheckbox).toBeVisible();
  });

  test('should update dot size value when slider is moved', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');
    await effectSelect.selectOption('halftone');

    const dotSizeSlider = page.locator('#dot-size-slider');
    const dotSizeValue = page.locator('#dot-size-value');

    // Move slider to a specific value
    await dotSizeSlider.fill('12');

    // Verify the value display updates
    await expect(dotSizeValue).toContainText('12');
  });

  test('should be able to select clustering effect', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');
    await effectSelect.selectOption('clustering');

    // Verify effect was selected
    await expect(effectSelect).toHaveValue('clustering');

    // Effect params container should exist (even if empty)
    const effectParams = page.locator('#effect-params');
    await expect(effectParams).toBeVisible();
  });

  test('should show kaleidoscope controls when selected', async ({ page }) => {
    const effectSelect = page.locator('#effect-select');
    await effectSelect.selectOption('kaleidoscope');

    // Check for segments slider
    const segmentsSlider = page.locator('#segments-slider');
    await expect(segmentsSlider).toBeVisible();

    // Check for rotation speed slider
    const rotationSlider = page.locator('#rotation-slider');
    await expect(rotationSlider).toBeVisible();
  });

  test('should have play/pause button disabled initially', async ({ page }) => {
    const playPauseBtn = page.locator('#play-pause-btn');
    await expect(playPauseBtn).toBeDisabled();
  });

  test('should have seek slider disabled initially', async ({ page }) => {
    const seekSlider = page.locator('#seek-slider');
    await expect(seekSlider).toBeDisabled();
  });

  test('should display initial time as 0:00', async ({ page }) => {
    const timeDisplay = page.locator('#time-display');
    await expect(timeDisplay).toContainText('0:00');
  });
});

test.describe('CyberVision Video Player - With Mock Video', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');

    // Mock the video API endpoint
    await page.route('**/api/video?path=*', async route => {
      // Return a mock video response
      await route.fulfill({
        status: 200,
        contentType: 'video/mp4',
        body: Buffer.from([]) // Empty buffer for testing
      });
    });
  });

  test('should attempt to load video when path is provided', async ({ page }) => {
    const videoPathInput = page.locator('#video-path');
    const loadVideoBtn = page.locator('#load-video-btn');

    await videoPathInput.fill('/test/video.mp4');
    await loadVideoBtn.click();

    // Should show loading status
    const statusMessage = page.locator('#status-message');
    await expect(statusMessage).toBeVisible();
  });
});
