// Mock WebGL renderer for testing
export async function initWebGL() {
  return {
    renderPassthrough: () => {},
    renderHalftone: () => {},
    renderMosaic: () => {},
    renderClustering: () => {},
    renderEdges: () => {},
    renderChromatic: () => {},
    renderGlitch: () => {},
    renderThermal: () => {},
    renderKaleidoscope: () => {},
    cleanup: () => {}
  };
}
