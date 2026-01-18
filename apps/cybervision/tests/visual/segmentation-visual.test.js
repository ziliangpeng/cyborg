import { test, expect } from '@playwright/test';

test.describe('Segmentation Visual Tests', () => {
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

  test('should not produce all-black output', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    // Wait for app to be available
    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    const result = await page.evaluate(async () => {
      const app = window.cyberVisionApp;

      // Create test input
      const mockVideo = document.createElement('canvas');
      mockVideo.width = 640;
      mockVideo.height = 480;
      const ctx = mockVideo.getContext('2d');
      ctx.fillStyle = 'red';
      ctx.fillRect(0, 0, 640, 480);

      // Create mask - half the image is foreground (255), half is background (0)
      const maskData = new Uint8Array(256 * 256);
      for (let i = 0; i < 256 * 256; i++) {
        maskData[i] = i < (256 * 256 / 2) ? 255 : 0;
      }

      await app.renderer.setupPipeline(mockVideo, 8);
      app.renderer.renderSegmentation(mockVideo, 'blur', 10, maskData, false, false);

      // Check output is not all black
      const canvas = document.querySelector('#canvas');
      const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      let nonBlackPixels = 0;
      let totalPixels = 0;
      for (let i = 0; i < imageData.data.length; i += 4) {
        totalPixels++;
        // Check if pixel is not black (r, g, b > threshold)
        if (imageData.data[i] > 10 || imageData.data[i+1] > 10 || imageData.data[i+2] > 10) {
          nonBlackPixels++;
        }
      }
      return { nonBlackPixels, totalPixels };
    });

    // At least 10% of pixels should be non-black (input is red, so should show red pixels)
    const nonBlackPercentage = (result.nonBlackPixels / result.totalPixels) * 100;
    expect(nonBlackPercentage).toBeGreaterThan(10);
  });

  test('should render different output for blur vs blackout modes', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    const result = await page.evaluate(async () => {
      const app = window.cyberVisionApp;

      // Create test input with gradient
      const mockVideo = document.createElement('canvas');
      mockVideo.width = 640;
      mockVideo.height = 480;
      const ctx = mockVideo.getContext('2d');
      const gradient = ctx.createLinearGradient(0, 0, 640, 0);
      gradient.addColorStop(0, 'red');
      gradient.addColorStop(1, 'blue');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 640, 480);

      // Create mask - half foreground, half background
      const maskData = new Uint8Array(256 * 256);
      for (let i = 0; i < 256 * 256; i++) {
        maskData[i] = i < (256 * 256 / 2) ? 255 : 0;
      }

      await app.renderer.setupPipeline(mockVideo, 8);

      // Render with blur mode
      app.renderer.renderSegmentation(mockVideo, 'blur', 10, maskData, false, false);
      const canvas = document.querySelector('#canvas');
      const blurImageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      // Render with blackout mode
      app.renderer.renderSegmentation(mockVideo, 'blackout', 10, maskData, false, false);
      const blackoutImageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      // Compare outputs - they should be different
      let differentPixels = 0;
      for (let i = 0; i < blurImageData.data.length; i++) {
        if (Math.abs(blurImageData.data[i] - blackoutImageData.data[i]) > 10) {
          differentPixels++;
        }
      }

      const totalValues = blurImageData.data.length;
      return { differentPixels, totalValues };
    });

    // At least 5% of values should be different between blur and blackout modes
    const differencePercentage = (result.differentPixels / result.totalValues) * 100;
    expect(differencePercentage).toBeGreaterThan(5);
  });

  test('should handle mask data correctly', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    const result = await page.evaluate(async () => {
      const app = window.cyberVisionApp;

      // Create test input
      const mockVideo = document.createElement('canvas');
      mockVideo.width = 640;
      mockVideo.height = 480;
      const ctx = mockVideo.getContext('2d');
      ctx.fillStyle = 'green';
      ctx.fillRect(0, 0, 640, 480);

      // Create mask with all foreground (255)
      const allForegroundMask = new Uint8Array(256 * 256).fill(255);

      // Create mask with all background (0)
      const allBackgroundMask = new Uint8Array(256 * 256).fill(0);

      await app.renderer.setupPipeline(mockVideo, 8);

      // Render with all foreground mask (should show original)
      app.renderer.renderSegmentation(mockVideo, 'blur', 10, allForegroundMask, false, false);
      const canvas = document.querySelector('#canvas');
      const foregroundImageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      // Render with all background mask (should be different)
      app.renderer.renderSegmentation(mockVideo, 'blur', 10, allBackgroundMask, false, false);
      const backgroundImageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      // Count non-black pixels in each case
      let foregroundNonBlack = 0;
      let backgroundNonBlack = 0;

      for (let i = 0; i < foregroundImageData.data.length; i += 4) {
        if (foregroundImageData.data[i] > 10 || foregroundImageData.data[i+1] > 10 || foregroundImageData.data[i+2] > 10) {
          foregroundNonBlack++;
        }
        if (backgroundImageData.data[i] > 10 || backgroundImageData.data[i+1] > 10 || backgroundImageData.data[i+2] > 10) {
          backgroundNonBlack++;
        }
      }

      return { foregroundNonBlack, backgroundNonBlack };
    });

    // All foreground should have significantly more non-black pixels than all background
    expect(result.foregroundNonBlack).toBeGreaterThan(result.backgroundNonBlack * 2);
  });

  test('should not produce NaN or invalid pixel values', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#gpuStatus')).toContain('WebGPU', { timeout: 5000 });

    await page.waitForFunction(() => window.cyberVisionApp != null && window.cyberVisionApp.renderer != null, { timeout: 5000 });

    const result = await page.evaluate(async () => {
      const app = window.cyberVisionApp;

      const mockVideo = document.createElement('canvas');
      mockVideo.width = 640;
      mockVideo.height = 480;
      const ctx = mockVideo.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, 640, 480);

      const maskData = new Uint8Array(256 * 256).fill(128);

      await app.renderer.setupPipeline(mockVideo, 8);
      app.renderer.renderSegmentation(mockVideo, 'blur', 10, maskData, false, false);

      const canvas = document.querySelector('#canvas');
      const imageData = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);

      // Check for NaN or out-of-range values
      let invalidValues = 0;
      for (let i = 0; i < imageData.data.length; i++) {
        const val = imageData.data[i];
        if (isNaN(val) || val < 0 || val > 255) {
          invalidValues++;
        }
      }

      return { invalidValues };
    });

    expect(result.invalidValues).toBe(0);
  });
});
