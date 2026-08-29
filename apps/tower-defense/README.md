# Tower Defense

A minimal, tasteful tower defense game — vanilla JS + Canvas, part of the cyborg apps family.

## Play

```bash
pnpm install        # from the cyborg root
pnpm --filter tower-defense dev    # open http://localhost:5174
```

**Controls:** `1 2 3` pick tower · click a free cell to build · click a tower to upgrade/sell · `enter` calls the next wave · `esc` cancels selection.

Survive 15 waves. Enemies leak through the path and cost lives. Three towers: Arrow (fast single-target), Frost (slows), Cannon (splash). Upgrades to level 3; selling refunds 60%.

## Testing (headless)

```bash
pnpm --filter tower-defense test      # vitest: 16 unit tests on the pure logic core
node tests/ui.e2e.js                  # Playwright + system Chrome: build/wave/clear/lose/restart flows + screenshots
```

Game rules live in `src/logic/game.js` (pure, DOM-free, deterministic — same pattern as `apps/snake-game`).
