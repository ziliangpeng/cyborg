// Canvas renderer. Draws interpolated state; knows nothing about game rules.

export function createRenderer(canvas, cellSize) {
  const ctx = canvas.getContext('2d');

  function draw(state, alpha, time) {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    drawGrid(state);
    drawFood(state, time);
    drawSnake(state, alpha);
  }

  function drawGrid(state) {
    ctx.fillStyle = 'rgba(235, 240, 220, 0.05)';
    for (let y = 0; y < state.height; y++) {
      for (let x = 0; x < state.width; x++) {
        ctx.beginPath();
        ctx.arc((x + 0.5) * cellSize, (y + 0.5) * cellSize, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function drawFood(state, time) {
    if (!state.food) return;
    const cx = (state.food.x + 0.5) * cellSize;
    const cy = (state.food.y + 0.5) * cellSize;
    const pulse = 1 + 0.09 * Math.sin(time / 280);

    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, cellSize * 1.5);
    g.addColorStop(0, 'rgba(255, 122, 92, 0.30)');
    g.addColorStop(1, 'rgba(255, 122, 92, 0)');
    ctx.fillStyle = g;
    ctx.fillRect(cx - cellSize * 1.5, cy - cellSize * 1.5, cellSize * 3, cellSize * 3);

    ctx.fillStyle = '#ff7a5c';
    ctx.beginPath();
    ctx.arc(cx, cy, cellSize * 0.26 * pulse, 0, Math.PI * 2);
    ctx.fill();

    // tiny highlight
    ctx.fillStyle = 'rgba(255, 235, 225, 0.85)';
    ctx.beginPath();
    ctx.arc(cx - cellSize * 0.07, cy - cellSize * 0.08, cellSize * 0.06, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawSnake(state, alpha) {
    const n = state.snake.length;
    const prev = state.prevSnake && state.prevSnake.length === n ? state.prevSnake : state.snake;

    // interpolated centers
    const pts = state.snake.map((seg, i) => ({
      x: (seg.x + (prev[i].x - seg.x) * (1 - alpha) + 0.5) * cellSize,
      y: (seg.y + (prev[i].y - seg.y) * (1 - alpha) + 0.5) * cellSize,
    }));

    for (let i = n - 1; i >= 0; i--) {
      const t = n === 1 ? 0 : i / (n - 1);
      // taper toward the tail, head slightly larger
      const size = cellSize * (0.82 - 0.18 * t) + (i === 0 ? cellSize * 0.06 : 0);
      const r = size * 0.38;

      const fade = 1 - 0.45 * t;
      const R = Math.round(0xcf * fade + (i === 0 ? 16 : 0));
      const G = Math.round(0xe3 * fade + (i === 0 ? 10 : 0));
      const B = Math.round(0xb8 * fade);

      ctx.fillStyle = `rgb(${R}, ${G}, ${B})`;
      roundRect(ctx, pts[i].x - size / 2, pts[i].y - size / 2, size, size, r);
      ctx.fill();
    }

    drawEyes(state, pts[0]);
  }

  function drawEyes(state, head) {
    const d = state.dir;
    // eye offsets: along direction of travel + perpendicular
    const table = {
      up:    { ax: 0,  ay: -0.16, px: 1, py: 0 },
      down:  { ax: 0,  ay: 0.16,  px: 1, py: 0 },
      left:  { ax: -0.16, ay: 0,  px: 0, py: 1 },
      right: { ax: 0.16,  ay: 0,  px: 0, py: 1 },
    };
    const o = table[d] ?? table.right;
    const r = cellSize * 0.075;
    const sep = cellSize * 0.16;

    for (const sign of [-1, 1]) {
      const ex = head.x + o.ax * cellSize + o.px * sep * sign;
      const ey = head.y + o.ay * cellSize + o.py * sep * sign;
      ctx.fillStyle = '#10130e';
      ctx.beginPath();
      ctx.arc(ex, ey, r, 0, Math.PI * 2);
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
