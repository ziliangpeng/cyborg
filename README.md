# Snake

A minimal, tasteful Snake game for the browser. Vanilla JS + Canvas, no framework.

## Play

```bash
npm install --include=dev
npm run dev          # open http://localhost:5173
```

**Controls:** arrows / WASD to move · space to pause · enter to restart after game over.

## Design

- 21×21 board, warm charcoal palette, Instrument Serif title, IBM Plex Mono numerals
- Smooth interpolated movement between ticks, tapered snake body, pulsing food
- Best score persists in localStorage; speed ramps up as you eat

## Testing (headless — no browser UI needed)

```bash
npm run test         # vitest: 19 unit tests on pure game logic
node tests/ui.spec.js  # Playwright + system Chrome: full integration, screenshots to ./shots
```

Game rules live in `src/logic/game.js` (pure, DOM-free, deterministic via seeded RNG).
Rendering (`src/ui/render.js`), input, and the game loop are kept separate so logic
tests run in milliseconds and UI checks run headless.
