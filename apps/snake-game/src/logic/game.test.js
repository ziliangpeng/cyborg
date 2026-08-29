import { describe, it, expect } from 'vitest';
import { createGame, tick, setDirection, start, spawnFood, makeRng } from './game.js';

function runningGame(opts = {}) {
  const s = createGame({ width: 11, height: 11, seed: 42, ...opts });
  s.food = { x: 50, y: 50 }; // unreachable cell on an 11x11 board -> no accidental food
  start(s);
  return s;
}

describe('createGame', () => {
  it('starts a snake in the center moving right', () => {
    const s = createGame({ width: 21, height: 21, seed: 1 });
    expect(s.snake[0]).toEqual({ x: 10, y: 10 });
    expect(s.dir).toBe('right');
    expect(s.status).toBe('idle');
    expect(s.score).toBe(0);
  });

  it('spawns food on a free cell', () => {
    const s = createGame({ seed: 1 });
    const cells = new Set(s.snake.map((p) => `${p.x},${p.y}`));
    expect(cells.has(`${s.food.x},${s.food.y}`)).toBe(false);
    expect(s.food.x).toBeGreaterThanOrEqual(0);
    expect(s.food.x).toBeLessThan(21);
  });
});

describe('movement', () => {
  it('moves right by one cell per tick', () => {
    const s = runningGame();
    const x0 = s.snake[0].x;
    tick(s);
    expect(s.snake[0].x).toBe(x0 + 1);
  });

  it('keeps a constant length when not eating', () => {
    const s = runningGame();
    for (let i = 0; i < 5; i++) tick(s);
    expect(s.snake).toHaveLength(3);
  });

  it('follows the head with its body (no gaps)', () => {
    const s = runningGame();
    tick(s);
    const [h, m, t] = s.snake;
    expect(Math.abs(h.x - m.x) + Math.abs(h.y - m.y)).toBe(1);
    expect(Math.abs(m.x - t.x) + Math.abs(m.y - t.y)).toBe(1);
  });
});

describe('directions', () => {
  it('turns up', () => {
    const s = runningGame();
    setDirection(s, 'up');
    tick(s);
    expect(s.snake[0].y).toBe(s.snake[1].y - 1);
  });

  it('ignores 180-degree reversal', () => {
    const s = runningGame();
    setDirection(s, 'left'); // opposite of 'right'
    tick(s);
    expect(s.dir).toBe('right');
    expect(s.snake[0].x).toBe(s.snake[1].x + 1);
  });

  it('ignores unknown directions', () => {
    const s = runningGame();
    setDirection(s, 'diagonal');
    tick(s);
    expect(s.dir).toBe('right');
  });

  it('queues a second turn so quick zig-zags work', () => {
    const s = runningGame(); // snake starts at (5,5) on an 11x11 board
    setDirection(s, 'up');
    setDirection(s, 'left');
    tick(s); // moves up
    expect(s.snake[0]).toEqual({ x: 5, y: 4 });
    tick(s); // moves left
    expect(s.snake[0]).toEqual({ x: 4, y: 4 });
  });

  it('rejects reversing via the queue too', () => {
    const s = runningGame();
    setDirection(s, 'up');
    setDirection(s, 'down'); // reverse of queued 'up'
    tick(s);
    expect(s.dir).toBe('up');
  });
});

describe('food & scoring', () => {
  it('grows and scores when eating food', () => {
    const s = runningGame({ seed: 7 });
    // force food directly ahead of the head
    const head = s.snake[0];
    s.food = { x: head.x + 1, y: head.y };
    const len = s.snake.length;
    tick(s);
    expect(s.score).toBe(1);
    expect(s.snake).toHaveLength(len + 1);
    expect(s.food).not.toBeNull();
    expect(s.food).not.toEqual({ x: head.x + 1, y: head.y });
  });

  it('never spawns food under the snake', () => {
    const rng = makeRng(123);
    for (let i = 0; i < 50; i++) rng();
    const s = runningGame({ seed: 123 });
    for (let i = 0; i < 30; i++) {
      const occ = new Set(s.snake.map((p) => `${p.x},${p.y}`));
      if (s.food) expect(occ.has(`${s.food.x},${s.food.y}`)).toBe(false);
      if (s.status === 'over') break;
      tick(s);
    }
  });
});

describe('game over', () => {
  it('dies on the right wall', () => {
    const s = runningGame({ width: 5, height: 5 });
    s.snake = [{ x: 4, y: 2 }, { x: 3, y: 2 }, { x: 2, y: 2 }];
    tick(s);
    expect(s.status).toBe('over');
  });

  it('dies on the top wall', () => {
    const s = runningGame({ width: 5, height: 5 });
    s.snake = [{ x: 2, y: 0 }, { x: 2, y: 1 }, { x: 2, y: 2 }];
    setDirection(s, 'up');
    tick(s);
    expect(s.status).toBe('over');
  });

  it('dies when running into its own body', () => {
    const s = runningGame({ width: 7, height: 7 });
    // U shape; the head turning up lands on a body cell (not the tail)
    s.snake = [
      { x: 3, y: 3 }, { x: 3, y: 4 }, { x: 4, y: 4 },
      { x: 4, y: 3 }, { x: 4, y: 2 }, { x: 3, y: 2 }, { x: 2, y: 2 },
    ];
    setDirection(s, 'up');
    tick(s);
    expect(s.status).toBe('over');
  });

  it('does NOT die on the tail cell that is about to move away', () => {
    const s = runningGame({ width: 9, height: 9 });
    // straight line moving right: tail will vacate as head advances
    s.snake = [{ x: 4, y: 4 }, { x: 3, y: 4 }, { x: 2, y: 4 }];
    s.dir = 'right';
    tick(s);
    expect(s.status).toBe('running');
    expect(s.snake[0]).toEqual({ x: 5, y: 4 });
  });

  it('freezes after game over', () => {
    const s = runningGame({ width: 5, height: 5 });
    s.snake = [{ x: 4, y: 2 }, { x: 3, y: 2 }, { x: 2, y: 2 }];
    tick(s);
    const snapshot = JSON.stringify(s.snake);
    tick(s);
    expect(JSON.stringify(s.snake)).toBe(snapshot);
  });
});

describe('win condition', () => {
  it('returns null food when the board is full', () => {
    const s = runningGame({ width: 3, height: 3 });
    // snake fills 8 of 9 cells; the last free cell is (1,1) directly below the head
    s.snake = [
      { x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 }, { x: 0, y: 2 },
      { x: 1, y: 2 }, { x: 2, y: 2 }, { x: 2, y: 1 }, { x: 2, y: 0 },
    ];
    s.dir = 'right';
    s.food = { x: 1, y: 1 };
    setDirection(s, 'down');
    tick(s);
    expect(s.status).toBe('running');
    expect(s.snake).toHaveLength(9); // whole board
    expect(s.food).toBeNull(); // board full, no free cell
  });
});

describe('determinism', () => {
  it('same seed -> same food sequence', () => {
    const a = createGame({ seed: 99 });
    const b = createGame({ seed: 99 });
    expect(a.food).toEqual(b.food);
    for (let i = 0; i < 5; i++) {
      tick(start(a)); tick(start(b));
      a.food = b.food = { x: -99, y: -99 }; // keep both alive
      // re-spawn deterministically
      a.rng = makeRng(99 + i); b.rng = makeRng(99 + i);
      a.food = spawnFood(a); b.food = spawnFood(b);
      expect(a.food).toEqual(b.food);
    }
  });
});
