import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const shadersDir = path.resolve(__dirname, '../../webroot/static/shaders');

describe('Shader file existence', () => {
  describe('WebGPU shaders (.wgsl)', () => {
    const wgslShaders = [
      'halftone.wgsl',
      'clustering.wgsl',
      'edges.wgsl',
      'mosaic.wgsl',
      'chromatic.wgsl',
      'glitch.wgsl',
      'thermal.wgsl',
      'pixelsort.wgsl',
      'pixelsort-segment.wgsl',
      'kaleidoscope.wgsl',
      'rotate.wgsl'
    ];

    wgslShaders.forEach(shaderFile => {
      it(`should have ${shaderFile}`, () => {
        const shaderPath = path.join(shadersDir, shaderFile);
        expect(fs.existsSync(shaderPath), `Shader file ${shaderFile} not found at ${shaderPath}`).toBe(true);
      });
    });
  });

  describe('WebGL shaders (.frag.glsl)', () => {
    const glslShaders = [
      'halftone.frag.glsl',
      'clustering.frag.glsl',
      'edges.frag.glsl',
      'mosaic.frag.glsl',
      'chromatic.frag.glsl',
      'glitch.frag.glsl',
      'thermal.frag.glsl',
      'kaleidoscope.frag.glsl',
      'passthrough.frag.glsl'
    ];

    glslShaders.forEach(shaderFile => {
      it(`should have ${shaderFile}`, () => {
        const shaderPath = path.join(shadersDir, shaderFile);
        expect(fs.existsSync(shaderPath), `Shader file ${shaderFile} not found at ${shaderPath}`).toBe(true);
      });
    });
  });

  describe('Shared vertex shader', () => {
    it('should have common.vert.glsl', () => {
      const shaderPath = path.join(shadersDir, 'common.vert.glsl');
      expect(fs.existsSync(shaderPath), `Vertex shader not found at ${shaderPath}`).toBe(true);
    });
  });

  describe('Shader content validation', () => {
    it('should have non-empty WebGPU shaders', () => {
      const clusteringShader = path.join(shadersDir, 'clustering.wgsl');
      const content = fs.readFileSync(clusteringShader, 'utf8');
      expect(content.length).toBeGreaterThan(0);
      // Check for WGSL syntax (vertex or compute shaders)
      const hasVertexOrCompute = content.includes('@vertex') || content.includes('@compute');
      expect(hasVertexOrCompute).toBe(true);
    });

    it('should have non-empty WebGL fragment shaders', () => {
      const halftoneShader = path.join(shadersDir, 'halftone.frag.glsl');
      const content = fs.readFileSync(halftoneShader, 'utf8');
      expect(content.length).toBeGreaterThan(0);
      expect(content).toContain('void main()');
    });

    it('should have valid vertex shader', () => {
      const vertexShader = path.join(shadersDir, 'common.vert.glsl');
      const content = fs.readFileSync(vertexShader, 'utf8');
      expect(content.length).toBeGreaterThan(0);
      expect(content).toContain('void main()');
      expect(content).toContain('gl_Position');
    });
  });
});
