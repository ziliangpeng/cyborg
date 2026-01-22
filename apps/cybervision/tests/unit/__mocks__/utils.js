// Mock utils for testing
export function calculateFPS(frameCount, elapsedMs) {
  if (elapsedMs <= 0) return 0;
  return Math.round((frameCount / elapsedMs) * 1000);
}

export function hexToRGB(hex) {
  const cleanHex = hex.replace('#', '');
  const r = parseInt(cleanHex.substring(0, 2), 16) / 255;
  const g = parseInt(cleanHex.substring(2, 4), 16) / 255;
  const b = parseInt(cleanHex.substring(4, 6), 16) / 255;
  return [r, g, b];
}

export function algorithmFromEffectName(effectName, useTrueColors = false) {
  return useTrueColors ? `${effectName}-true` : effectName;
}

export function parseResolution(resolutionStr) {
  const parts = resolutionStr.split('x');
  return { width: parseInt(parts[0], 10), height: parseInt(parts[1], 10) };
}

export function average(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}
