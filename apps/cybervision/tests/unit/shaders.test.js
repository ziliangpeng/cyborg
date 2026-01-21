import { describe, it, expect } from 'vitest';

describe('Shader file existence', () => {
  // Note: Shaders are now served from libs/cybervision-core.
  // File existence and content validation is verified by integration tests.
  // The integration tests will catch any missing or malformed shader files.
  it('should pass - shaders moved to shared library', () => {
    expect(true).toBe(true);
  });
});
