// Canvas renderer for tower defense. Pure drawing; no game rules here.

export function createRenderer(canvas, cell) {
  const ctx = canvas.getContext('2d');

  const COLORS = {
    arrow: '#cfe3b8',
    frost: '#a8c5d8',
    cannon: '#e0b184',
    enemy: '#ff7a5c',
    fast: '#ffb08c',
    tank: '#d95f43',
  };

  function draw(state, ui, time) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawGrid(state);
    drawPath(state);
    drawRangePreview(state, ui);
    drawTowers(state, ui);
    drawEnemies(state, ui);
    drawProjectiles(state);
    drawEffects(ui, time);
  }

  function drawGrid(state) {
    ctx.fillStyle = 'rgba(235, 240, 220, 0.045)';
    for (let y = 0; y < state.grid.rows; y++) {
      for (let x = 0; x < state.grid.cols; x++) {
        ctx.beginPath();
        ctx.arc((x + 0.5) * cell, (y + 0.5) * cell, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawPath(state) {
    const pts = state.path.pts;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo((pts[0].x + 0.5) * cell, (pts[0].y + 0.5) * cell);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo((pts[i].x + 0.5) * cell, (pts[i].y + 0.5) * cell);
    }
    ctx.strokeStyle = 'rgba(233, 231, 220, 0.055)';
    ctx.lineWidth = cell * 0.78;
    ctx.stroke();
    // dashed flow line drifting toward the exit
    ctx.strokeStyle = 'rgba(233, 231, 220, 0.035)';
    ctx.lineWidth = cell * 0.78;
    ctx.setLineDash([cell * 0.35, cell * 0.45]);
    ctx.lineDashOffset = -((state.time * 18) % 1000);
    ctx.stroke();
    ctx.setLineDash([]);
    // exit marker
    const exit = pts[pts.length - 1];
    ctx.fillStyle = 'rgba(255, 122, 92, 0.5)';
    ctx.beginPath();
    ctx.arc((exit.x + 0.5) * cell, (exit.y + 0.5) * cell, cell * 0.12, 0, Math.PI * 2);
    ctx.fill();
  }

  function canPlaceLike(state, g) {
    if (g.gx < 0 || g.gy < 0 || g.gx >= state.grid.cols || g.gy >= state.grid.rows) return false;
    if (state.pathSet.has(`${g.gx},${g.gy}`)) return false;
    return !state.towers.some((t) => t.gx === g.gx && t.gy === g.gy);
  }

  function dashedCircle(x, y, r, color) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawRangePreview(state, ui) {
    if (ui.hover && ui.selectedType && ui.selectedType !== null && canPlaceLike(state, ui.hover)) {
      dashedCircle(
        (ui.hover.gx + 0.5) * cell, (ui.hover.gy + 0.5) * cell,
        ui.selectedRange * cell, 'rgba(207, 227, 184, 0.5)'
      );
    }
    if (ui.selectedTower) {
      dashedCircle(
        (ui.selectedTower.gx + 0.5) * cell, (ui.selectedTower.gy + 0.5) * cell,
        ui.selectedRange * cell, 'rgba(207, 227, 184, 0.55)'
      );
    }
  }

  function drawTowers(state, ui) {
    for (const t of state.towers) {
      const cx = (t.gx + 0.5) * cell;
      const cy = (t.gy + 0.5) * cell;
      const size = cell * 0.62;
      const isSel = ui.selectedTower && ui.selectedTower.id === t.id;

      ctx.fillStyle = COLORS[t.type];
      roundRect(ctx, cx - size / 2, cy - size / 2, size, size, size * 0.3);
      ctx.fill();
      if (isSel) {
        ctx.strokeStyle = 'rgba(233, 231, 220, 0.9)';
        ctx.lineWidth = 2;
        roundRect(ctx, cx - size / 2, cy - size / 2, size, size, size * 0.3);
        ctx.stroke();
      }
      // barrel pointing at the last target angle
      if (t.aimAngle !== undefined) {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(t.aimAngle);
        ctx.fillStyle = 'rgba(16, 19, 14, 0.5)';
        roundRect(ctx, cell * 0.08, -3, cell * 0.3, 6, 3);
        ctx.fill();
        ctx.restore();
      }
      // level pips
      ctx.fillStyle = 'rgba(16, 19, 14, 0.55)';
      for (let i = 0; i < t.level; i++) {
        ctx.beginPath();
        ctx.arc(cx - (t.level - 1) * 3.5 + i * 7, cy + size * 0.32, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawEnemies(state, ui) {
    for (const e of state.enemies) {
      const p = ui.enemyPos(e);
      const cx = (p.x + 0.5) * cell;
      const cy = (p.y + 0.5) * cell;
      const base = e.type === 'tank' ? 0.36 : e.type === 'fast' ? 0.24 : 0.3;
      const r = cell * base;

      ctx.fillStyle = COLORS[e.type] || COLORS.enemy;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      if (state.time < e.slowUntil) {
        ctx.strokeStyle = 'rgba(168, 197, 216, 0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
        ctx.stroke();
      }
      const w = cell * 0.6;
      ctx.fillStyle = 'rgba(16, 19, 14, 0.6)';
      ctx.fillRect(cx - w / 2, cy - r - 8, w, 3.5);
      ctx.fillStyle = COLORS.enemy;
      ctx.fillRect(cx - w / 2, cy - r - 8, w * (e.hp / e.maxHp), 3.5);
    }
  }

  function drawProjectiles(state) {
    for (const pr of state.projectiles) {
      const x = (pr.x + 0.5) * cell;
      const y = (pr.y + 0.5) * cell;
      if (pr.kind === 'cannon') {
        ctx.fillStyle = '#e0b184';
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = pr.kind === 'frost' ? COLORS.frost : COLORS.arrow;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawEffects(ui, time) {
    for (const fx of ui.effects) {
      const age = (time - fx.t0) / fx.dur;
      if (age >= 1) continue;
      const t = 1 - Math.pow(1 - age, 2);
      const r = fx.r0 + (fx.r1 - fx.r0) * t;
      ctx.beginPath();
      ctx.arc((fx.x + 0.5) * cell, (fx.y + 0.5) * cell, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 160, 100, ${(1 - age) * 0.8})`;
      ctx.lineWidth = 2.5 * (1 - age);
      ctx.stroke();
      ctx.fillStyle = `rgba(255, 122, 92, ${(1 - age) * 0.22})`;
      ctx.fill();
    }
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  return { draw };
}
