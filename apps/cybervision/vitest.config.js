import { defineConfig } from 'vitest/config';
import * as path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/unit/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['lcov', 'text'],
      include: ['webroot/static/**/*.js'],
      // When Bazel runs coverage, write to the coverage output directory
      // Otherwise, write to ./coverage for local development
      reportsDirectory: process.env.COVERAGE_OUTPUT_FILE
        ? path.dirname(process.env.COVERAGE_OUTPUT_FILE)
        : './coverage',
    },
  },
});
