import { test, expect } from '@playwright/test';

test.describe('CyberVision Video Player', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the page successfully', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('CyberVision Player');
    await expect(page.locator('#chooseFileBtn')).toBeVisible();
    await expect(page.locator('#dropZone')).toBeVisible();
    // New UI uses effect buttons instead of select
    await expect(page.locator('.effect-btn').first()).toBeVisible();
  });

  test('should have all effect options available', async ({ page }) => {
    // Check for effect buttons across both tabs
    const artisticTab = page.locator('.tab-button[data-tab="artistic"]');
    const distortionTab = page.locator('.tab-button[data-tab="distortion"]');

    await expect(artisticTab).toBeVisible();
    await expect(distortionTab).toBeVisible();

    // Check artistic effects
    await artisticTab.click();
    await expect(page.locator('.effect-btn[data-effect="halftone"]')).toBeVisible();
    await expect(page.locator('.effect-btn[data-effect="clustering"]')).toBeVisible();
    await expect(page.locator('.effect-btn[data-effect="kaleidoscope"]')).toBeVisible();

    // Check distortion effects
    await distortionTab.click();
    await expect(page.locator('.effect-btn[data-effect="original"]')).toBeVisible();
    await expect(page.locator('.effect-btn[data-effect="edges"]')).toBeVisible();
  });

  test('should have file picker button visible', async ({ page }) => {
    await expect(page.locator('#chooseFileBtn')).toBeVisible();
    await expect(page.locator('#chooseFileBtn')).toContainText('Choose Video File');
  });

  test('should have drop zone visible', async ({ page }) => {
    await expect(page.locator('#dropZone')).toBeVisible();
  });

  test('should trigger file input when button is clicked', async ({ page }) => {
    const fileInput = page.locator('#videoFile');
    await expect(fileInput).toBeHidden(); // Hidden but present

    // Verify the button triggers the file input
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#chooseFileBtn');
    const fileChooser = await fileChooserPromise;
    expect(fileChooser).toBeTruthy();
  });

  test('should display filename after file selection', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#chooseFileBtn');
    const fileChooser = await fileChooserPromise;

    // Select a file (Playwright can set file even without real file)
    await fileChooser.setFiles({
      name: 'test-video.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from([])
    });

    await expect(page.locator('#currentFile')).toContainText('test-video.mp4');
  });

  test('should show drag-over visual feedback', async ({ page }) => {
    const dropZone = page.locator('#dropZone');

    // Use page.evaluate to dispatch drag events with proper dataTransfer
    await page.evaluate(() => {
      const dropZoneEl = document.getElementById('dropZone');
      const dragOverEvent = new Event('dragover', { bubbles: true });
      dragOverEvent.preventDefault = () => {};
      dropZoneEl.dispatchEvent(dragOverEvent);
    });
    await expect(dropZone).toHaveClass(/drag-over/);

    // Simulate dragleave event
    await page.evaluate(() => {
      const dropZoneEl = document.getElementById('dropZone');
      const dragLeaveEvent = new Event('dragleave', { bubbles: true });
      dropZoneEl.dispatchEvent(dragLeaveEvent);
    });
    await expect(dropZone).not.toHaveClass(/drag-over/);
  });

  test('should change effect selection', async ({ page }) => {
    // Click on halftone effect button (in artistic tab)
    const artisticTab = page.locator('.tab-button[data-tab="artistic"]');
    await artisticTab.click();

    const halftoneBtn = page.locator('.effect-btn[data-effect="halftone"]');
    await halftoneBtn.click();

    // Verify halftone controls appear
    const halftoneControls = page.locator('#halftoneControls');
    await expect(halftoneControls).toBeVisible();
  });

  test('should show halftone controls when halftone effect is selected', async ({ page }) => {
    // Click on halftone effect button
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="halftone"]').click();

    // Check for dot size slider (new ID: dotSizeSlider)
    const dotSizeSlider = page.locator('#dotSizeSlider');
    await expect(dotSizeSlider).toBeVisible();

    // Check for random colors checkbox (new ID: randomColorCheckbox)
    const randomColorsCheckbox = page.locator('#randomColorCheckbox');
    await expect(randomColorsCheckbox).toBeVisible();
  });

  test('should update dot size value when slider is moved', async ({ page }) => {
    // Select halftone effect
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="halftone"]').click();

    const dotSizeSlider = page.locator('#dotSizeSlider');
    const dotSizeValue = page.locator('#dotSizeValue');

    // Move slider to a specific value
    await dotSizeSlider.fill('12');

    // Verify the value display updates
    await expect(dotSizeValue).toContainText('12');
  });

  test('should be able to select clustering effect', async ({ page }) => {
    // Click on clustering effect button
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="clustering"]').click();

    // Verify clustering controls appear
    const clusteringControls = page.locator('#clusteringControls');
    await expect(clusteringControls).toBeVisible();
  });

  test('should show kaleidoscope controls when selected', async ({ page }) => {
    // Click on kaleidoscope effect button
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="kaleidoscope"]').click();

    // Check for segments slider (new ID: segmentsSlider)
    const segmentsSlider = page.locator('#segmentsSlider');
    await expect(segmentsSlider).toBeVisible();

    // Check for rotation speed slider (new ID: rotationSpeedSlider)
    const rotationSlider = page.locator('#rotationSpeedSlider');
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

  test('should show segmentation controls when segmentation effect is selected', async ({ page }) => {
    // Click on segmentation effect button
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="segmentation"]').click();

    // Verify segmentation controls appear
    const segmentationControls = page.locator('#segmentationControls');
    await expect(segmentationControls).toBeVisible();

    // Check for segmentation mode dropdown (new ID: segmentationMode)
    const segmentationMode = page.locator('#segmentationMode');
    await expect(segmentationMode).toBeVisible();

    // Verify the dropdown has blackout selected by default
    await expect(segmentationMode).toHaveValue('blackout');

    // Check for blur radius slider
    const blurRadiusSlider = page.locator('#segmentationBlurRadius');
    await expect(blurRadiusSlider).toBeVisible();

    // Check for soft edges and glow checkboxes
    const softEdgesCheckbox = page.locator('#segmentationSoftEdges');
    const glowCheckbox = page.locator('#segmentationGlow');
    await expect(softEdgesCheckbox).toBeVisible();
    await expect(glowCheckbox).toBeVisible();
  });

  test('should not have critical console errors when selecting segmentation', async ({ page }) => {
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        // Filter out expected model loading messages
        if (!text.includes('segmentation model') && !text.includes('Not Found')) {
          errors.push(text);
        }
      }
    });

    // Click on segmentation effect button
    await page.locator('.tab-button[data-tab="artistic"]').click();
    await page.locator('.effect-btn[data-effect="segmentation"]').click();

    // Wait a bit for any async errors to appear
    await page.waitForTimeout(500);

    // Should not have critical errors (ONNX Runtime loading errors, etc.)
    const criticalErrors = errors.filter(err =>
      err.includes('ort') ||
      err.includes('ONNX') ||
      err.includes('is not defined')
    );
    expect(criticalErrors).toHaveLength(0);
  });
});

test.describe('CyberVision Video Player - With Local Video', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load video when file is selected', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#chooseFileBtn');
    const fileChooser = await fileChooserPromise;

    // Select a file
    await fileChooser.setFiles({
      name: 'test-video.mp4',
      mimeType: 'video/mp4',
      buffer: Buffer.from([])
    });

    // Should show loading status
    const statusMessage = page.locator('#status');
    await expect(statusMessage).toBeVisible();
  });
});

test.describe('CyberVision Video Player - External Dependencies', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('ONNX Runtime library is loaded', async ({ page }) => {
    const ortLoaded = await page.evaluate(() => typeof window.ort !== 'undefined');
    expect(ortLoaded).toBe(true);
  });

  test('server serves /libs/ route for ML inference', async ({ request }) => {
    const response = await request.get('/libs/cybervision-core/ml-inference.js');
    expect(response.status()).toBe(200);
  });

  test('server serves /models/ route', async ({ request }) => {
    const response = await request.get('/models/segmentation.onnx');
    // 200 if model exists, 404 is acceptable if model not downloaded yet
    // What matters is the route exists (not a 500 or "Cannot GET" error)
    expect([200, 404]).toContain(response.status());
  });
});
