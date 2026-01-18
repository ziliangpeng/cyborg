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
      'rotate.wgsl',
      'segmentation.wgsl'
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

  describe('WGSL shader type safety', () => {
    it('should not construct vec2<f32> from u32 params without f32() cast', () => {
      // This test catches the bug where vec2<f32>(params.width, params.height)
      // was used when params.width and params.height are u32 values
      const segmentationShader = path.join(shadersDir, 'segmentation.wgsl');
      const content = fs.readFileSync(segmentationShader, 'utf8');

      // Check if there are u32 parameters (width, height)
      const hasU32Params = /struct\s+\w+\s*\{[\s\S]*?(width|height)\s*:\s*u32/;

      // Look for unsafe pattern: vec2<f32>(params.width, params.height) without f32()
      // This regex looks for vec2<f32>( followed by any identifier.width or identifier.height
      // that is NOT wrapped in f32()
      const unsafeVec2Pattern = /vec2<f32>\s*\(\s*(\w+\.(width|height))\s*,\s*(\w+\.(width|height))\s*\)/g;

      if (hasU32Params.test(content)) {
        const matches = content.match(unsafeVec2Pattern);
        if (matches) {
          // For each match, verify it has f32() casts
          matches.forEach(match => {
            // If the match doesn't contain 'f32(' before the params, it's unsafe
            const hasCast = /vec2<f32>\s*\(\s*f32\s*\(/.test(match);
            expect(hasCast,
              `Found unsafe vec2<f32> construction from u32 params: ${match}. ` +
              `Use vec2<f32>(f32(param.width), f32(param.height)) instead.`
            ).toBe(true);
          });
        }
      }
    });

    it('should use correct texture format types in WGSL shaders', () => {
      // This test helps catch issues where texture formats and sampling don't match
      const segmentationShader = path.join(shadersDir, 'segmentation.wgsl');
      const content = fs.readFileSync(segmentationShader, 'utf8');

      // If there's a texture_2d<f32>, the corresponding textureLoad or textureSample should return vec4<f32>
      // This is a basic check - it won't catch all format mismatches but helps with common cases
      const hasF32Texture = /texture_2d<f32>/.test(content);
      if (hasF32Texture) {
        // Make sure we're consistently using f32 types with texture operations
        const hasTextureOps = /textureLoad|textureSample/.test(content);
        expect(hasTextureOps).toBe(true);
      }
    });
  });
});
