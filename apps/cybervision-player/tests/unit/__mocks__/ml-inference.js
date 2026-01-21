// Mock ML inference for testing
export class PortraitSegmentation {
  constructor() {}
  async loadModel() {}
  async segment() {
    return new ImageData(1, 1);
  }
}
