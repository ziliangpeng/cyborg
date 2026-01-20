import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '/lib/webgpu-renderer.js': path.resolve(__dirname, 'tests/unit/__mocks__/webgpu-renderer.js'),
      '/libs/cybervision-core/ml-inference.js': path.resolve(__dirname, 'tests/unit/__mocks__/ml-inference.js'),
      '/lib': path.resolve(__dirname, '../../libs/cybervision-core'),
      '/libs/cybervision-core': path.resolve(__dirname, '../../libs/cybervision-core'),
      '/shaders': path.resolve(__dirname, '../../libs/cybervision-core/shaders'),
    }
  },
  test: {
    environment: 'happy-dom',
    globals: true,
    include: ['tests/unit/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      include: ['webroot/static/**/*.js'],
      exclude: ['webroot/static/shaders/**']
    }
  }
});
