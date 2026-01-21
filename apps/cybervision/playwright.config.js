import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:8000',
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'ci',
      testMatch: /.*\.test\.js/,
      testIgnore: [/.*webgpu.*\.test\.js/, /unit\//, /visual\//],
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--disable-software-rasterizer=false',
            '--headless=new'
          ],
        },
      },
    },
    {
      name: 'local',
      testMatch: /.*\.test\.js/,
      testIgnore: [/unit\//, /visual\//],
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--enable-features=Vulkan'],
        },
      },
    },
    {
      name: 'visual',
      testMatch: /visual\/.*\.test\.js/,
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--enable-features=Vulkan'],
        },
      },
    },
  ],

  webServer: {
    // Use cybervision binary if available (Bazel tests), otherwise use main.py from repo root (pnpm tests)
    command: process.env.BAZEL_TEST ? './cybervision --port 8000' : 'python3 -m apps.cybervision.main --port 8000',
    cwd: process.env.BAZEL_TEST ? undefined : '../..',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
});
