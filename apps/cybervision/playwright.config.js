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
      testIgnore: [/.*webgpu.*\.test\.js/, /unit\//],
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
      testIgnore: /unit\//,
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: ['--enable-features=Vulkan'],
        },
      },
    },
  ],

  webServer: {
    command: 'python3 -m http.server 8000 -d webroot',
    url: 'http://localhost:8000',
    reuseExistingServer: !process.env.CI,
  },
});
