/**
 * Shared test helpers for CyberVision E2E tests
 */

/**
 * Map of effect values to their tab names
 */
const effectToTab = {
  'halftone': 'artistic',
  'duotone': 'artistic',
  'dither': 'artistic',
  'clustering': 'artistic',
  'mosaic': 'artistic',
  'kaleidoscope': 'artistic',
  'pixelsort': 'artistic',
  'segmentation': 'artistic',
  'original': 'distortion',
  'edges': 'distortion',
  'twirl': 'distortion',
  'chromatic': 'distortion',
  'glitch': 'distortion',
  'thermal': 'distortion'
};

/**
 * Switches to the tab containing a specific effect
 * @param {import('@playwright/test').Page} page - Playwright page object
 * @param {string} effectValue - The value of the effect (e.g., 'halftone')
 */
export async function switchToEffectTab(page, effectValue) {
  const tabName = effectToTab[effectValue];
  if (tabName) {
    const tabButton = page.locator(`button[data-tab="${tabName}"]`);
    const tabContent = page.locator(`#tab-${tabName}`);
    const isActive = await tabButton.evaluate(el => el.classList.contains('active'));

    if (!isActive) {
      await tabButton.click();
      await tabContent.waitFor({ state: 'visible', timeout: 2000 });
    }
  }
}

export async function waitForAppInit(page) {
  await expect(page.locator('#gpuStatus')).toHaveText(/WebGL|WebGPU|No support/, { timeout: 10000 });
}
