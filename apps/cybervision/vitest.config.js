import { defineConfig } from 'vitest/config';
import * as path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '/lib/webgpu-renderer.js': path.resolve(__dirname, 'tests/unit/__mocks__/webgpu-renderer.js'),
      '/lib/webgl-renderer.js': path.resolve(__dirname, 'tests/unit/__mocks__/webgl-renderer.js'),
      '/lib/ml-inference.js': path.resolve(__dirname, 'tests/unit/__mocks__/ml-inference.js'),
      '/lib/utils.js': path.resolve(__dirname, 'tests/unit/__mocks__/utils.js'),
      '/lib/histogram.js': path.resolve(__dirname, '../../libs/cybervision-core/histogram.js'),
      '/libs/cybervision-core/ml-inference.js': path.resolve(__dirname, 'tests/unit/__mocks__/ml-inference.js'),
      '/lib': path.resolve(__dirname, '../../libs/cybervision-core'),
      '/libs/cybervision-core': path.resolve(__dirname, '../../libs/cybervision-core'),
      '/shaders': path.resolve(__dirname, '../../libs/cybervision-core/shaders'),
    }
  },
  test: {
    environmentOptions: {
      happyDOM: {
        url: 'http://127.0.0.1:3000',
      },
    },
    globals: true,
    environment: 'happy-dom', // Changed from 'node'
    include: ['tests/unit/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['lcov', 'text'],
      include: ['webroot/static/**/*.js'],
      exclude: ['webroot/static/shaders/**'], // Added from player
      // When Bazel runs coverage, write to the coverage output directory
      // Otherwise, write to ./coverage for local development
      reportsDirectory: process.env.COVERAGE_OUTPUT_FILE
        ? path.dirname(process.env.COVERAGE_OUTPUT_FILE)
        : './coverage',
    },
  },
});
