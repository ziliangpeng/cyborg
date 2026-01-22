// Mock WebGPU renderer for testing
export async function initGPU() {
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
    renderPixelSort: () => {},
    renderSegmentation: () => {},
    updateBackgroundImage: () => {},
    updateDotSize: () => {},
    setupPipeline: async () => {},
    cleanup: () => {}
  };
}

export class WebGPURenderer {
  constructor() {}
  async init() {}
  updateDotSize() {}
  updateColorCount() {}
  updateSegments() {}
  setEffect() {}
  renderFrame() {}
}
