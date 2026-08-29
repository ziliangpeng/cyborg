// Pure game logic — no DOM, no rendering. Fully deterministic given seed.

export const DIRS = {
  up: { x: 0, y: -1 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
  right: { x: 1, y: 0 },
};

// Deterministic PRNG (mulberry32) so tests can seed food placement.
export function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createGame({ width = 21, height = 21, seed = Date.now(), length = 3 } = {}) {
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const snake = [];
  for (let i = 0; i < length; i++) snake.push({ x: cx - i, y: cy });

  const state = {
    width, height,
    snake,
    dir: 'right',
    status: 'idle', // idle | running | over
    score: 0,
    food: null,
    rng: makeRng(seed),
    _pendingDirs: [],
  };
  state.food = spawnFood(state);
  return state;
}

function occupied(state) {
  const set = new Set();
  for (const s of state.snake) set.add(s.y * state.width + s.x);
  return set;
}

export function spawnFood(state) {
  const { width, height, rng } = state;
  const occ = occupied(state);
  const free = [];
  for (let i = 0; i < width * height; i++) if (!occ.has(i)) free.push(i);
  if (free.length === 0) return null; // board full -> win
  const idx = free[Math.floor(rng() * free.length)];
  return { x: idx % width, y: Math.floor(idx / width) };
}

const opposite = { up: 'down', down: 'up', left: 'right', right: 'left' };

// Queue a direction change. Reversals are ignored.
export function setDirection(state, dir) {
  if (!(dir in DIRS)) return state;
  const last = state._pendingDirs.length
    ? state._pendingDirs[state._pendingDirs.length - 1]
    : state.dir;
  if (dir === last || dir === opposite[last]) return state;
  state._pendingDirs.push(dir);
  return state;
}

// Advance one fixed step. Mutates state; returns it for convenience.
export function tick(state) {
  if (state.status !== 'running') return state;

  if (state._pendingDirs.length) state.dir = state._pendingDirs.shift();
  const d = DIRS[state.dir];
  const head = state.snake[0];
  const nx = head.x + d.x;
  const ny = head.y + d.y;

  // Wall collision
  if (nx < 0 || ny < 0 || nx >= state.width || ny >= state.height) {
    state.status = 'over';
    return state;
  }

  // Self collision — tail cell is safe unless we're growing this tick
  const willGrow = state.food && nx === state.food.x && ny === state.food.y;
  const body = willGrow ? state.snake : state.snake.slice(0, -1);
  if (body.some((s) => s.x === nx && s.y === ny)) {
    state.status = 'over';
    return state;
  }

  state.snake.unshift({ x: nx, y: ny });
  if (willGrow) {
    state.score += 1;
    state.food = spawnFood(state);
  } else {
    state.snake.pop();
  }
  return state;
}

export function start(state) {
  if (state.status === 'idle') state.status = 'running';
  return state;
}
