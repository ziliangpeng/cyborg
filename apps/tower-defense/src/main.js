import {
  createGame, tick, startWave, placeTower, canPlace, upgradeTower, upgradeCost,
  sellTower, towerStats, TOWERS, MAX_WAVES,
} from './logic/game.js';
import { createRenderer } from './ui/render.js';

const CELL = 42;
const STEP = 1 / 60;
const canvas = document.getElementById('board');
const moneyEl = document.getElementById('money');
const livesEl = document.getElementById('lives');
const waveEl = document.getElementById('wave');
const waveBtn = document.getElementById('wave-btn');
const overlay = document.getElementById('overlay');
const overlayTitle = document.getElementById('overlay-title');
const overlayHint = document.getElementById('overlay-hint');
const buildBar = document.getElementById('build-bar');
const panel = document.getElementById('tower-panel');
const panelName = document.getElementById('panel-name');
const panelLevel = document.getElementById('panel-level');
const upgradeBtn = document.getElementById('upgrade-btn');
const sellBtn = document.getElementById('sell-btn');

const renderer = createRenderer(canvas, CELL);

let state = createGame();
let selectedType = 'arrow';
let hover = null;
let selectedTower = null;
let uiAlpha = 0;
const effects = [];

// ---------- position interpolation for smooth enemy motion ----------

function posAt(progress) {
  const { pts, cum } = state.path;
  if (progress <= 0) return { x: pts[0].x, y: pts[0].y };
  if (progress >= cum[cum.length - 1]) {
    const last = pts[pts.length - 1];
    return { x: last.x, y: last.y };
  }
  let lo = 0, hi = cum.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] <= progress) lo = mid; else hi = mid;
  }
  const segLen = cum[hi] - cum[lo] || 1;
  const t = (progress - cum[lo]) / segLen;
  return { x: pts[lo].x + (pts[hi].x - pts[lo].x) * t, y: pts[lo].y + (pts[hi].y - pts[lo].y) * t };
}

function enemyPos(e) {
  if (!e.prev) return e.cur ?? { x: -99, y: -99 };
  return {
    x: e.prev.x + (e.cur.x - e.prev.x) * (1 - uiAlpha),
    y: e.prev.y + (e.cur.y - e.prev.y) * (1 - uiAlpha),
  };
}

function addEffect(gx, gy, r0, r1, dur = 400) {
  effects.push({ x: gx, y: gy, t0: performance.now(), dur, r0, r1 });
}

// ---------- HUD ----------

function updateHud() {
  moneyEl.textContent = state.money;
  livesEl.textContent = state.lives;
  waveEl.textContent = `${state.wave}/${MAX_WAVES}`;
  waveBtn.disabled = state.phase !== 'build' || state.status !== 'running';
  waveBtn.textContent = state.wave === 0 ? 'Call wave 1' : `Call wave ${state.wave + 1}`;

  for (const card of buildBar.querySelectorAll('.tower-card')) {
    const type = card.dataset.type;
    card.classList.toggle('selected', type === selectedType);
    card.classList.toggle('unaffordable', state.money < TOWERS[type].cost);
  }

  if (selectedTower) {
    const still = state.towers.find((t) => t.id === selectedTower.id);
    if (!still) {
      selectedTower = null;
      panel.classList.add('hidden');
    } else {
      const cost = upgradeCost(still);
      panelName.textContent = TOWERS[still.type].name;
      panelLevel.textContent = `lv ${still.level}/3`;
      upgradeBtn.textContent = cost === null ? 'Max level' : `Upgrade ${cost}`;
      upgradeBtn.disabled = cost === null || state.money < cost;
      sellBtn.textContent = `Sell +${Math.round(still.spent * 0.6)}`;
    }
  }
}

function showOverlay(title, hint) {
  overlayTitle.textContent = title;
  overlayHint.textContent = hint;
  overlay.classList.remove('hidden');
}

// ---------- input ----------

function gameToCanvas(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

canvas.addEventListener('mousemove', (e) => {
  const p = gameToCanvas(e);
  hover = { gx: Math.floor(p.x / CELL), gy: Math.floor(p.y / CELL) };
});

canvas.addEventListener('mouseleave', () => { hover = null; });

canvas.addEventListener('click', (e) => {
  if (state.status !== 'running') return;
  const p = gameToCanvas(e);
  const gx = Math.floor(p.x / CELL);
  const gy = Math.floor(p.y / CELL);

  const hitTower = state.towers.find((t) => t.gx === gx && t.gy === gy);
  if (hitTower) {
    selectedTower = hitTower;
    panel.classList.remove('hidden');
    updateHud();
    return;
  }
  selectedTower = null;
  panel.classList.add('hidden');

  if (selectedType && state.phase === 'build') {
    if (placeTower(state, selectedType, gx, gy)) {
      const t = state.towers[state.towers.length - 1];
      addEffect(t.gx, t.gy, CELL * 0.2, CELL * 0.75);
      updateHud();
    }
  }
});

buildBar.addEventListener('click', (e) => {
  const card = e.target.closest('.tower-card');
  if (!card) return;
  selectedType = card.dataset.type;
  selectedTower = null;
  panel.classList.add('hidden');
  updateHud();
});

waveBtn.addEventListener('click', () => {
  if (startWave(state)) { overlay.classList.add('hidden'); updateHud(); }
});

upgradeBtn.addEventListener('click', () => {
  if (selectedTower && upgradeTower(state, selectedTower.id)) {
    addEffect(selectedTower.gx, selectedTower.gy, CELL * 0.3, CELL * 0.95);
    updateHud();
  }
});

sellBtn.addEventListener('click', () => {
  if (selectedTower && sellTower(state, selectedTower.id)) {
    selectedTower = null;
    panel.classList.add('hidden');
    updateHud();
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === '1') selectedType = 'arrow';
  else if (e.key === '2') selectedType = 'frost';
  else if (e.key === '3') selectedType = 'cannon';
  else if (e.key === 'Escape') { selectedTower = null; panel.classList.add('hidden'); }
  else if ((e.key === 'Enter' || e.key === ' ') && state.phase === 'build' && state.status === 'running') {
    e.preventDefault();
    if (startWave(state)) overlay.classList.add('hidden');
  } else if (e.key === 'Enter' && state.status !== 'running') {
    state = createGame();
    selectedTower = null;
    panel.classList.add('hidden');
    showOverlay('Build your defenses', 'Pick a tower, click a free cell · press enter to call the first wave');
    updateHud();
    return;
  } else {
    return;
  }
  selectedTower = null;
  if (['1', '2', '3'].includes(e.key)) panel.classList.add('hidden');
  updateHud();
});

// ---------- main loop ----------

let last = performance.now();
let acc = 0;

function frame(now) {
  const dtMs = Math.min(100, now - last);
  last = now;
  uiAlpha = Math.min(1, acc / STEP);

  if (state.status === 'running') {
    acc += dtMs / 1000;
    let guard = 0;
    while (acc >= STEP && guard++ < 8) {
      acc -= STEP;
      // remember positions before stepping (for interpolation)
      for (const en of state.enemies) {
        en.prev = posAt(en.progress);
      }
      // track cannon shells so we can spawn explosion effects where they land
      const cannonBefore = new Map(
        state.projectiles.filter((p) => p.kind === 'cannon').map((p) => [p, { x: p.aimX, y: p.aimY }])
      );
      tick(state, STEP);
      // update current positions
      for (const en of state.enemies) {
        en.cur = posAt(en.progress);
        if (!en.prev) en.prev = { ...en.cur };
      }
      // detect landed cannon shells
      for (const [pr, aim] of cannonBefore) {
        if (!state.projectiles.includes(pr)) addEffect(aim.x - 0.5, aim.y - 0.5, CELL * 0.3, CELL * 1.3, 420);
      }
      if (state.status !== 'running') {
        if (state.status === 'over') showOverlay('Overwhelmed', `You held out for ${state.wave} wave${state.wave === 1 ? '' : 's'} · press enter to try again`);
        else if (state.status === 'won') showOverlay('Victory', `All ${MAX_WAVES} waves repelled · press enter to play again`);
        break;
      }
      if (state.phase === 'build' && state.wave > 0 && state.wave < MAX_WAVES) {
        // wave cleared banner handled by overlay only at end; keep playing
      }
    }
  }

  const selRange = selectedTower
    ? towerStats(selectedTower).range
    : selectedType ? TOWERS[selectedType].range : 0;

  renderer.draw(state, {
    hover, selectedType, selectedRange: selRange, selectedTower, effects,
    enemyPos,
  }, now);

  updateHud();
  requestAnimationFrame(frame);
}

requestAnimationFrame((t) => { last = t; frame(t); });

if (typeof window !== 'undefined') {
  window.__game = {
    get state() { return state; },
    get ui() { return { selectedType, selectedTower, hover }; },
    select: (t) => { selectedType = t; },
  };
}
