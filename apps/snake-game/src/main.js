import { createGame, tick, setDirection, start } from './logic/game.js';
import { createRenderer } from './ui/render.js';

const GRID = 21;                 // cells per side
const CANVAS = 630;              // px
const CELL = CANVAS / GRID;
const BASE_STEP = 150;           // ms per tick at score 0
const MIN_STEP = 70;             // fastest step
const SPEEDUP = 2.2;             // ms shaved per food

const canvas = document.getElementById('board');
const scoreEl = document.getElementById('score');
const bestEl = document.getElementById('best');
const overlay = document.getElementById('overlay');
const overlayTitle = document.getElementById('overlay-title');
const overlayHint = document.getElementById('overlay-hint');

const renderer = createRenderer(canvas, CELL);

let state = createGame({ width: GRID, height: GRID });
let best = Number(localStorage.getItem('snake-best') ?? 0);
bestEl.textContent = best;

let stepMs = BASE_STEP;
let acc = 0;
let last = performance.now();
let renderAlpha = 0;

function stepDuration() {
  return Math.max(MIN_STEP, BASE_STEP - state.score * SPEEDUP);
}

function snapshotPrev() {
  state.prevSnake = state.snake.map((s) => ({ ...s }));
}

function advance() {
  snapshotPrev();
  tick(state);
  // When the snake grows, the new tail cell has no previous position.
  // Pad prevSnake with the old tail so interpolation keeps running and
  // the growth animates as a slide instead of a one-frame snap.
  while (state.prevSnake.length < state.snake.length) {
    state.prevSnake.push({ ...state.prevSnake[state.prevSnake.length - 1] });
  }
  if (state.food) {
    stepMs = stepDuration();
  }
  if (state.status === 'over') onGameOver();
  scoreEl.textContent = state.score;
}

function onGameOver() {
  if (state.score > best) {
    best = state.score;
    localStorage.setItem('snake-best', best);
    bestEl.textContent = best;
  }
  overlayTitle.textContent = state.score === 0 ? 'Oops.' : `Score ${state.score}`;
  overlayHint.textContent = 'Press enter to play again';
  overlay.classList.remove('hidden');
}

function reset() {
  state = createGame({ width: GRID, height: GRID });
  stepMs = BASE_STEP;
  overlayTitle.textContent = 'Ready?';
  overlayHint.textContent = 'Press an arrow key to play';
  scoreEl.textContent = '0';
}

function frame(now) {
  const dt = Math.min(100, now - last);
  last = now;

  if (state.status === 'running') {
    acc += dt;
    while (acc >= stepMs) {
      acc -= stepMs;
      advance();
      if (state.status !== 'running') { acc = 0; break; }
    }
    renderAlpha = state.status === 'running' ? acc / stepMs : 1;
  } else {
    renderAlpha = 1;
  }

  renderer.draw(state, renderAlpha, now);
  requestAnimationFrame(frame);
}

const KEY_TO_DIR = {
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  w: 'up', s: 'down', a: 'left', d: 'right',
  W: 'up', S: 'down', A: 'left', D: 'right',
};

window.addEventListener('keydown', (e) => {
  const dir = KEY_TO_DIR[e.key];
  if (dir) {
    e.preventDefault();
    if (state.status === 'idle') {
      start(state);
      overlay.classList.add('hidden');
    } else if (state.status === 'over') {
      return;
    }
    setDirection(state, dir);
  } else if (e.key === ' ') {
    e.preventDefault();
    if (state.status === 'running') {
      state.status = 'paused';
      overlayTitle.textContent = 'Paused';
      overlayHint.textContent = 'Press space to resume';
      overlay.classList.remove('hidden');
    } else if (state.status === 'paused') {
      state.status = 'running';
      overlay.classList.add('hidden');
    }
  } else if (e.key === 'Enter' && state.status === 'over') {
    reset();
    // starting immediately on a fresh board via next arrow press
  }
});

// test hook
if (typeof window !== 'undefined') {
  window.__game = {
    get state() { return state; },
    advance,
  };
}

requestAnimationFrame((t) => { last = t; frame(t); });
