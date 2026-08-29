// Pure tower-defense logic. No DOM, no rendering, fully deterministic.

export const GRID = { cols: 15, rows: 11 };

// Waypoints in cell coords (floats). The path enters left, winds S-shaped, exits right.
const WAYPOINTS = [
  { x: -1, y: 2 }, { x: 11, y: 2 }, { x: 11, y: 5 }, { x: 3, y: 5 },
  { x: 3, y: 8 }, { x: 16, y: 8 },
];

export function buildPath(waypoints = WAYPOINTS) {
  const pts = [];
  for (let i = 0; i < waypoints.length - 1; i++) {
    const a = waypoints[i], b = waypoints[i + 1];
    const len = Math.abs(b.x - a.x) + Math.abs(b.y - a.y);
    const steps = Math.ceil(len);
    for (let s = 0; s < steps; s++) {
      const t = s / steps;
      pts.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, seg: i });
    }
  }
  pts.push({ ...waypoints[waypoints.length - 1], seg: waypoints.length - 1 });
  // cumulative length along the path
  let acc = 0;
  const cum = pts.map((p, i) => {
    if (i > 0) acc += Math.abs(pts[i].x - pts[i - 1].x) + Math.abs(pts[i].y - pts[i - 1].y);
    return acc;
  });
  return { pts, cum, total: acc };
}

// Cells the path occupies (for build blocking), by rounding to nearest int cell.
export function pathCells(path = buildPath()) {
  const set = new Set();
  for (const p of path.pts) set.add(`${Math.round(p.x)},${Math.round(p.y)}`);
  // also fill straight segments between waypoint corners so no diagonal gaps
  const wps = WAYPOINTS;
  for (let i = 0; i < wps.length - 1; i++) {
    const a = wps[i], b = wps[i + 1];
    const dx = Math.sign(b.x - a.x), dy = Math.sign(b.y - a.y);
    let x = a.x, y = a.y;
    set.add(`${Math.round(x)},${Math.round(y)}`);
    while (x !== b.x || y !== b.y) {
      x += (x !== b.x ? dx : 0);
      y += (x === b.x && y !== b.y ? dy : 0);
      set.add(`${Math.round(x)},${Math.round(y)}`);
    }
  }
  return set;
}

export const TOWERS = {
  arrow:  { name: 'Arrow',  cost: 50,  dmg: 8,  rate: 2.2, range: 2.6 },
  frost:  { name: 'Frost',  cost: 70,  dmg: 4,  rate: 1.6, range: 2.3, slowFactor: 0.5, slowDur: 1.8 },
  cannon: { name: 'Cannon', cost: 110, dmg: 22, rate: 0.75, range: 2.9, splash: 1.2 },
};

export const ENEMIES = {
  normal: { hpMul: 1.0,  speed: 1.5, bounty: 6 },
  fast:   { hpMul: 0.65, speed: 2.6, bounty: 8 },
  tank:   { hpMul: 2.6,  speed: 0.9, bounty: 14 },
};

export const MAX_WAVES = 15;
export const START_MONEY = 130;
export const START_LIVES = 20;
export const SELL_REFUND = 0.6;

let nextId = 1;
export function resetIds() { nextId = 1; }

// Wave composition: deterministic from the wave number.
export function waveSpec(wave) {
  const count = Math.min(8 + wave * 2, 26);
  const hp = 22 * Math.pow(1.28, wave - 1);
  const comp = [];
  for (let i = 0; i < count; i++) {
    let type = 'normal';
    if (wave >= 3 && i % 4 === 3) type = 'fast';
    if (wave >= 5 && wave % 5 === 0 && i % 5 === 0) type = 'tank';
    comp.push(type);
  }
  return { count, hp, comp, interval: Math.max(0.45, 1.0 - wave * 0.04) };
}

export function createGame() {
  resetIds();
  const path = buildPath();
  return {
    grid: { ...GRID },
    path,
    pathSet: pathCells(path),
    towers: [],
    enemies: [],
    projectiles: [],
    money: START_MONEY,
    lives: START_LIVES,
    wave: 0,
    phase: 'build',        // build (between waves) | wave (spawning/fighting)
    spawnQueue: [],
    spawnTimer: 0,
    time: 0,
    status: 'running',     // running | over | won
  };
}

// --- placement ---

export function canPlace(state, gx, gy) {
  if (gx < 0 || gy < 0 || gx >= state.grid.cols || gy >= state.grid.rows) return false;
  if (state.pathSet.has(`${gx},${gy}`)) return false;
  return !state.towers.some((t) => t.gx === gx && t.gy === gy);
}

export function placeTower(state, type, gx, gy) {
  const def = TOWERS[type];
  if (!def || state.status !== 'running' || state.phase !== 'build') return false;
  if (state.money < def.cost || !canPlace(state, gx, gy)) return false;
  state.money -= def.cost;
  state.towers.push({
    id: nextId++, type, gx, gy, level: 1,
    cooldown: 0, spent: def.cost,
  });
  return true;
}

export function towerStats(tower) {
  const def = TOWERS[tower.type];
  const lv = tower.level - 1;
  return {
    dmg: def.dmg * Math.pow(1.5, lv),
    range: def.range * Math.pow(1.1, lv),
    rate: def.rate * Math.pow(1.12, lv),
    splash: def.splash ? def.splash * Math.pow(1.08, lv) : undefined,
  };
}

export function upgradeCost(tower) {
  if (tower.level >= 3) return null;
  return Math.round(TOWERS[tower.type].cost * 0.9 * tower.level);
}

export function upgradeTower(state, id) {
  const t = state.towers.find((t) => t.id === id);
  if (!t) return false;
  const cost = upgradeCost(t);
  if (cost === null || state.money < cost) return false;
  state.money -= cost;
  t.spent += cost;
  t.level += 1;
  return true;
}

export function sellTower(state, id) {
  const i = state.towers.findIndex((t) => t.id === id);
  if (i === -1) return false;
  state.money += Math.round(state.towers[i].spent * SELL_REFUND);
  state.towers.splice(i, 1);
  return true;
}

// --- waves ---

export function startWave(state) {
  if (state.status !== 'running' || state.phase !== 'build') return false;
  state.wave += 1;
  const spec = waveSpec(state.wave);
  state.spawnQueue = spec.comp.map((type, i) => ({
    type, hp: spec.hp * ENEMIES[type].hpMul, at: i * spec.interval,
  }));
  state.spawnTimer = 0;
  state.phase = 'wave';
  return true;
}

// --- combat tick (dt in seconds) ---

function posAt(path, progress) {
  const { pts, cum } = path;
  if (progress <= 0) return { x: pts[0].x, y: pts[0].y };
  if (progress >= cum[cum.length - 1]) {
    const last = pts[pts.length - 1];
    return { x: last.x, y: last.y };
  }
  // binary search
  let lo = 0, hi = cum.length - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (cum[mid] <= progress) lo = mid; else hi = mid;
  }
  const segLen = cum[hi] - cum[lo] || 1;
  const t = (progress - cum[lo]) / segLen;
  return {
    x: pts[lo].x + (pts[hi].x - pts[lo].x) * t,
    y: pts[lo].y + (pts[hi].y - pts[lo].y) * t,
  };
}

export function tick(state, dt) {
  if (state.status !== 'running' || dt <= 0) return state;
  state.time += dt;

  // spawning
  if (state.spawnQueue.length) {
    state.spawnTimer += dt;
    while (state.spawnQueue.length && state.spawnQueue[0].at <= state.spawnTimer) {
      const s = state.spawnQueue.shift();
      const e = ENEMIES[s.type];
      state.enemies.push({
        id: nextId++, type: s.type, hp: s.hp, maxHp: s.hp,
        speed: e.speed, bounty: e.bounty, progress: 0, slowUntil: 0, slowFactor: 1,
      });
    }
  }

  // enemies move
  for (const e of state.enemies) {
    const slowed = state.time < e.slowUntil;
    const speed = e.speed * (slowed ? e.slowFactor : 1);
    e.progress += speed * dt;
  }
  // leaks
  const pathEnd = state.path.total;
  for (let i = state.enemies.length - 1; i >= 0; i--) {
    if (state.enemies[i].progress >= pathEnd) {
      state.enemies.splice(i, 1);
      state.lives -= 1;
      if (state.lives <= 0) { state.lives = 0; state.status = 'over'; return state; }
    }
  }

  // towers shoot
  for (const t of state.towers) {
    t.cooldown -= dt;
    if (t.cooldown > 0) continue;
    const st = towerStats(t);
    const tx = t.gx + 0.5, ty = t.gy + 0.5;
    let best = null;
    for (const e of state.enemies) {
      const p = posAt(state.path, e.progress);
      const d = Math.hypot(p.x - tx, p.y - ty);
      if (d <= st.range && (!best || d < best.d)) best = { e, d, p };
    }
    if (!best) continue;
    t.cooldown = 1 / st.rate;
    const { type } = t;
    if (type === 'cannon') {
      // fired at the target's position; explodes there (no homing)
      state.projectiles.push({
        kind: 'cannon', x: tx, y: ty, aimX: best.p.x, aimY: best.p.y,
        speed: 7, dmg: st.dmg, splash: st.splash, ownerId: t.id,
      });
    } else {
      state.projectiles.push({
        kind: type, x: tx, y: ty, targetId: best.e.id,
        speed: 9, dmg: st.dmg, ownerId: t.id,
      });
    }
  }

  // projectiles
  for (let i = state.projectiles.length - 1; i >= 0; i--) {
    const pr = state.projectiles[i];
    if (pr.kind === 'cannon') {
      const d = Math.hypot(pr.aimX - pr.x, pr.aimY - pr.y);
      const step = pr.speed * dt;
      if (d <= step) {
        // explode
        for (const e of state.enemies) {
          const p = posAt(state.path, e.progress);
          if (Math.hypot(p.x - pr.aimX, p.y - pr.aimY) <= pr.splash) damage(state, e, pr.dmg);
        }
        state.projectiles.splice(i, 1);
      } else {
        pr.x += ((pr.aimX - pr.x) / d) * step;
        pr.y += ((pr.aimY - pr.y) / d) * step;
      }
    } else {
      const target = state.enemies.find((e) => e.id === pr.targetId);
      if (!target) { state.projectiles.splice(i, 1); continue; }
      const p = posAt(state.path, target.progress);
      const d = Math.hypot(p.x - pr.x, p.y - pr.y);
      const step = pr.speed * dt;
      if (d <= step) {
        damage(state, target, pr.dmg);
        if (pr.kind === 'frost') {
          target.slowUntil = state.time + TOWERS.frost.slowDur;
          target.slowFactor = TOWERS.frost.slowFactor;
        }
        state.projectiles.splice(i, 1);
      } else {
        pr.x += ((p.x - pr.x) / d) * step;
        pr.y += ((p.y - pr.y) / d) * step;
      }
    }
  }

  // deaths + bounty
  for (let i = state.enemies.length - 1; i >= 0; i--) {
    if (state.enemies[i].hp <= 0) {
      state.money += state.enemies[i].bounty;
      state.enemies.splice(i, 1);
    }
  }

  // wave complete?
  if (state.phase === 'wave' && !state.spawnQueue.length && !state.enemies.length) {
    state.phase = 'build';
    state.money += 25 + state.wave * 3; // wave-clear bonus
    if (state.wave >= MAX_WAVES) { state.status = 'won'; }
  }
  return state;
}

function damage(state, enemy, dmg) {
  enemy.hp -= dmg;
}
