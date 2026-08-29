// Playwright integration tests, run directly with `node tests/ui.e2e.js`.
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const BASE = 'http://localhost:5174';
const SHOT_DIR = 'shots';
mkdirSync(SHOT_DIR, { recursive: true });
async function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function startServer() {
  const proc = spawn('npx', ['vite', '--port', '5174', '--strictPort'], { stdio: 'pipe' });
  for (let i = 0; i < 60; i++) {
    try { const res = await fetch(BASE); if (res.ok) return proc; } catch {}
    await wait(250);
  }
  proc.kill();
  throw new Error('dev server failed to start');
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
  await page.waitForTimeout(800);

  // 1. board renders at expected size
  const box = await page.locator('#board').boundingBox();
  check('board is 630x462', box && box.width === 630 && box.height === 462, JSON.stringify(box));
  check('overlay visible at start', await page.locator('#overlay').isVisible());
  await page.screenshot({ path: `${SHOT_DIR}/01-build-phase.png` });

  // 2. build an arrow tower at a free cell via the card + click
  await page.click('.tower-card[data-type="arrow"]');
  // cell (12,0) is free per logic tests; board origin found from canvas box
  const cell = { x: box.x + 12 * 42 + 21, y: box.y + 0 * 42 + 21 };
  await page.mouse.click(cell.x, cell.y);
  let st = await page.evaluate(() => ({
    towers: window.__game.state.towers.length, money: window.__game.state.money,
  }));
  check('tower placed via click', st.towers === 1 && st.money === 80, JSON.stringify(st));

  // 3. call the wave, enemies spawn
  await page.keyboard.press('Enter');
  await page.waitForTimeout(1500);
  st = await page.evaluate(() => ({
    phase: window.__game.state.phase, enemies: window.__game.state.enemies.length, wave: window.__game.state.wave,
  }));
  check('wave started and enemies spawned', st.phase === 'wave' && st.enemies > 0, JSON.stringify(st));

  // 4. enemies die to the tower; wave clears back to build
  let cleared = false;
  for (let i = 0; i < 60; i++) {
    await page.waitForTimeout(1000);
    st = await page.evaluate(() => ({
      phase: window.__game.state.phase, status: window.__game.state.status, money: window.__game.state.money,
    }));
    if (st.phase === 'build') { cleared = true; break; }
    if (st.status !== 'running') break;
  }
  check('wave 1 cleared by one arrow tower', cleared && st.money > 80, JSON.stringify(st));

  // mid-action screenshot with a wave running: start wave 2 and capture
  await page.evaluate(() => { window.__game.select('cannon'); });
  await page.mouse.click(box.x + 10 * 42 + 21, box.y + 1 * 42 + 21); // affordable? money should be ~ 80+bounty+bonus
  await page.keyboard.press('Enter');
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${SHOT_DIR}/02-combat.png` });

  // 5. force game over: no defenses matter, drain lives
  await page.evaluate(() => {
    const s = window.__game.state;
    s.lives = 1;
    s.towers.length = 0;
  });
  await page.evaluate(() => {
    const s2 = window.__game.state;
    s2.towers.length = 0;
    s2.lives = 1;
    const end = s2.path.total - 1.5; // drop everyone near the exit
    for (const en of s2.enemies) en.progress = end;
  });
  await page.waitForTimeout(6000);
  st = await page.evaluate(() => ({ status: window.__game.state.status, lives: window.__game.state.lives }));
  check('game over reached', st.status === 'over' && st.lives === 0, JSON.stringify(st));

  // 6. restart with Enter
  await page.keyboard.press('Enter');
  await page.waitForTimeout(200);
  st = await page.evaluate(() => {
    const s = window.__game.state;
    return { status: s.status, money: s.money, wave: s.wave, towers: s.towers.length };
  });
  check('restart resets the game', st.money === 130 && st.wave === 0 && st.towers === 0, JSON.stringify(st));

  console.log(failures === 0 ? '\nALL E2E TESTS PASSED' : `\n${failures} FAILURES`);
} finally {
  if (server) server.kill();
  await browser.close();
}
process.exit(failures === 0 ? 0 : 1);
