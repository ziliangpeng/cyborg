// Mock WebGL renderer for testing
export async function initWebGL() {
  return {
    renderPassthrough: () => {},
    renderHalftone: () => {},
    renderMosaic: () => {},
    renderDuotone: () => {},
    renderDither: () => {},
    renderPosterize: () => {},
    renderClustering: () => {},
    renderEdges: () => {},
    renderTwirl: () => {},
    renderVignette: () => {},
    renderChromatic: () => {},
    renderGlitch: () => {},
    renderThermal: () => {},
    renderKaleidoscope: () => {},
    cleanup: () => {}
  };
}
