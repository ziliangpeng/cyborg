/* CyberVision - Utility functions */

/**
 * Maps effect names to algorithm strings for the clustering effect
 * @param {string} effectName - The base effect name
 * @param {boolean} useTrueColors - Whether to use true colors
 * @returns {string} The algorithm string
 */
export function algorithmFromEffectName(effectName, useTrueColors = false) {
  return useTrueColors ? `${effectName}-true` : effectName;
}

/**
 * Calculates FPS from frame count and elapsed time
 * @param {number} frameCount - Number of frames rendered
 * @param {number} elapsedMs - Time elapsed in milliseconds
 * @returns {number} Frames per second (rounded)
 */
export function calculateFPS(frameCount, elapsedMs) {
  if (elapsedMs <= 0) return 0;
  return Math.round((frameCount / elapsedMs) * 1000);
}

/**
 * Parses resolution string like "1920x1080"
 * @param {string} resolutionStr - Resolution string in format "WIDTHxHEIGHT"
 * @returns {{width: number, height: number}} Width and height as numbers
 */
export function parseResolution(resolutionStr) {
  const parts = resolutionStr.split('x');
  if (parts.length !== 2) {
    throw new Error('Invalid resolution format. Expected "WIDTHxHEIGHT"');
  }
  const width = parseInt(parts[0], 10);
  const height = parseInt(parts[1], 10);
  if (isNaN(width) || isNaN(height)) {
    throw new Error('Invalid resolution values. Width and height must be numbers');
  }
  return { width, height };
}

/**
 * Converts hex color to RGB array (0-1 range)
 * @param {string} hex - Hex color string (with or without #)
 * @returns {number[]} RGB array [r, g, b] with values 0-1
 */
export function hexToRGB(hex) {
  const cleanHex = hex.replace('#', '');
  if (cleanHex.length !== 6) {
    throw new Error('Invalid hex color. Expected 6 characters');
  }
  const r = parseInt(cleanHex.substring(0, 2), 16) / 255;
  const g = parseInt(cleanHex.substring(2, 4), 16) / 255;
  const b = parseInt(cleanHex.substring(4, 6), 16) / 255;
  return [r, g, b];
}

/**
 * Calculates average from an array of numbers
 * @param {number[]} values - Array of numbers
 * @returns {number} Average value
 */
export function average(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}
