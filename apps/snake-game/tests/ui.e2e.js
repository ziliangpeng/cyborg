// Minimal Playwright runner using the `playwright` package + system Chrome.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const BASE = 'http://localhost:5173';
const SHOT_DIR = 'shots';
mkdirSync(SHOT_DIR, { recursive: true });

async function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function startServer() {
  const proc = spawn('npx', ['vite', '--port', '5173', '--strictPort'], {
    cwd: process.cwd(),
    stdio: 'pipe',
  });
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(BASE);
      if (res.ok) return proc;
    } catch {}
    await wait(250);
  }
  proc.kill();
  throw new Error('vite dev server failed to start');
}

let failures = 0;
function check(name, cond, extra = '') {
  if (cond) console.log(`  ✓ ${name}`);
  else { failures++; console.error(`  ✗ ${name} ${extra}`); }
}

const browser = await chromium.launch({ channel: 'chrome', headless: true });
let server;

try {
  server = await startServer();
  const page = await browser.newPage({ viewport: { width: 960, height: 860 } });
  await page.goto(BASE);
  await page.waitForTimeout(800); // fonts/layout settle

  // 1. board renders
  const box = await page.locator('#board').boundingBox();
  check('board is visible with size', !!box && box.width === 630, JSON.stringify(box));

  // 2. initial overlay
  check('idle overlay shown', await page.locator('#overlay').isVisible());
  await page.screenshot({ path: `${SHOT_DIR}/01-idle.png` });

  // 3. start + move right
  await page.keyboard.press('ArrowRight');
  await page.waitForTimeout(150);
  const s1 = await page.evaluate(() => window.__game.state.status);
  check('game started after arrow key', s1 === 'running', `status=${s1}`);
  const x0 = await page.evaluate(() => window.__game.state.snake[0].x);
  await page.waitForTimeout(600);
  const x1 = await page.evaluate(() => window.__game.state.snake[0].x);
  check('snake moved right', x1 > x0, `x0=${x0} x1=${x1}`);

  // 4. eat food: place food ahead, wait, check score
  await page.evaluate(() => {
    const st = window.__game.state;
    const head = st.snake[0];
    st.food = { x: head.x + 2, y: head.y };
  });
  await page.waitForTimeout(700);
  const score = await page.locator('#score').textContent();
  check('score increments after eating', Number(score) >= 1, `score=${score}`);

  // mid-game screenshot
  await page.evaluate(() => {
    const st = window.__game.state;
    st.food = { x: st.snake[0].x + 3, y: st.snake[0].y - 1 };
  });
  await page.waitForTimeout(120);
  await page.screenshot({ path: `${SHOT_DIR}/02-playing.png` });

  // 5. force game over into the right wall
  await page.evaluate(() => {
    const st = window.__game.state;
    st.snake = [{ x: 19, y: 10 }, { x: 18, y: 10 }, { x: 17, y: 10 }];
    st.food = { x: 0, y: 0 };
  });
  await page.waitForTimeout(900);
  const over = await page.evaluate(() => window.__game.state.status);
  check('game over on wall collision', over === 'over', `status=${over}`);
  check('game-over overlay shown', await page.locator('#overlay').isVisible());
  await page.screenshot({ path: `${SHOT_DIR}/03-gameover.png` });

  // 6. restart with Enter
  await page.keyboard.press('Enter');
  await page.waitForTimeout(200);
  const st2 = await page.evaluate(() => window.__game.state);
  check('restart resets state', st2.status === 'idle' && st2.score === 0 && st2.snake.length === 3,
    JSON.stringify({ status: st2.status, score: st2.score, len: st2.snake.length }));

  console.log(failures === 0 ? '\nALL UI TESTS PASSED' : `\n${failures} FAILURES`);
} finally {
  if (server) server.kill();
  await browser.close();
}
process.exit(failures === 0 ? 0 : 1);
