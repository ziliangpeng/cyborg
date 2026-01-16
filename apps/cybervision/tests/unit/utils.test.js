import { describe, it, expect } from 'vitest';
import {
  algorithmFromEffectName,
  calculateFPS,
  parseResolution,
  hexToRGB,
  average
} from '../../webroot/static/utils.js';

describe('algorithmFromEffectName', () => {
  it('should return base algorithm when useTrueColors is false', () => {
    expect(algorithmFromEffectName('quantization-kmeans', false)).toBe('quantization-kmeans');
    expect(algorithmFromEffectName('quantization-median-cut', false)).toBe('quantization-median-cut');
  });

  it('should append -true when useTrueColors is true', () => {
    expect(algorithmFromEffectName('quantization-kmeans', true)).toBe('quantization-kmeans-true');
    expect(algorithmFromEffectName('quantization-median-cut', true)).toBe('quantization-median-cut-true');
  });

  it('should default to false for useTrueColors', () => {
    expect(algorithmFromEffectName('quantization-kmeans')).toBe('quantization-kmeans');
  });
});

describe('calculateFPS', () => {
  it('should calculate FPS correctly', () => {
    expect(calculateFPS(60, 1000)).toBe(60);
    expect(calculateFPS(120, 1000)).toBe(120);
    expect(calculateFPS(30, 500)).toBe(60);
  });

  it('should round FPS to nearest integer', () => {
    expect(calculateFPS(55, 1000)).toBe(55);
    expect(calculateFPS(55, 1001)).toBe(55); // 54.945... rounds to 55
  });

  it('should return 0 for zero or negative elapsed time', () => {
    expect(calculateFPS(60, 0)).toBe(0);
    expect(calculateFPS(60, -100)).toBe(0);
  });
});

describe('parseResolution', () => {
  it('should parse valid resolution strings', () => {
    expect(parseResolution('1920x1080')).toEqual({ width: 1920, height: 1080 });
    expect(parseResolution('1280x720')).toEqual({ width: 1280, height: 720 });
    expect(parseResolution('640x480')).toEqual({ width: 640, height: 480 });
  });

  it('should throw error for invalid format', () => {
    expect(() => parseResolution('1920-1080')).toThrow('Invalid resolution format');
    expect(() => parseResolution('1920')).toThrow('Invalid resolution format');
    expect(() => parseResolution('')).toThrow('Invalid resolution format');
  });

  it('should throw error for non-numeric values', () => {
    expect(() => parseResolution('abcxdef')).toThrow('Invalid resolution values');
    expect(() => parseResolution('1920xabc')).toThrow('Invalid resolution values');
  });
});

describe('hexToRGB', () => {
  it('should convert hex colors to RGB arrays', () => {
    expect(hexToRGB('#ffffff')).toEqual([1, 1, 1]);
    expect(hexToRGB('#000000')).toEqual([0, 0, 0]);
    expect(hexToRGB('#ff0000')).toEqual([1, 0, 0]);
    expect(hexToRGB('#00ff00')).toEqual([0, 1, 0]);
    expect(hexToRGB('#0000ff')).toEqual([0, 0, 1]);
  });

  it('should handle hex without # prefix', () => {
    expect(hexToRGB('ffffff')).toEqual([1, 1, 1]);
    expect(hexToRGB('ff0000')).toEqual([1, 0, 0]);
  });

  it('should convert intermediate values correctly', () => {
    const result = hexToRGB('#808080');
    expect(result[0]).toBeCloseTo(0.502, 2); // 128/255 ≈ 0.502
    expect(result[1]).toBeCloseTo(0.502, 2);
    expect(result[2]).toBeCloseTo(0.502, 2);
  });

  it('should throw error for invalid hex colors', () => {
    expect(() => hexToRGB('#fff')).toThrow('Invalid hex color');
    expect(() => hexToRGB('#fffffff')).toThrow('Invalid hex color');
    expect(() => hexToRGB('invalid')).toThrow('Invalid hex color');
  });
});

describe('average', () => {
  it('should calculate average of numbers', () => {
    expect(average([1, 2, 3, 4, 5])).toBe(3);
    expect(average([10, 20, 30])).toBe(20);
    expect(average([100])).toBe(100);
  });

  it('should handle decimal values', () => {
    expect(average([1.5, 2.5, 3.5])).toBeCloseTo(2.5);
  });

  it('should return 0 for empty array', () => {
    expect(average([])).toBe(0);
  });

  it('should return 0 for null or undefined', () => {
    expect(average(null)).toBe(0);
    expect(average(undefined)).toBe(0);
  });
});
