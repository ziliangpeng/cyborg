import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '/lib': path.resolve(__dirname, '../../libs/cybervision-core'),
      '/shaders': path.resolve(__dirname, '../../libs/cybervision-core/shaders'),
    }
  },
  test: {
    environment: 'jsdom',
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
