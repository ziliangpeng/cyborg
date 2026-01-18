/**
 * ML Inference Wrapper for ONNX Runtime Web
 * Handles model loading, inference, and hardware acceleration
 */

class MLInference {
  constructor() {
    this.session = null;
    this.modelPath = null;
    this.isLoaded = false;
    this.isLoading = false;
    this.executionProvider = null;
  }

  /**
   * Detect available execution providers
   * Returns { webgpu: boolean, wasm: boolean }
   */
  async detectCapabilities() {
    const capabilities = {
      webgpu: false,
      wasm: true // WASM is always available
    };

    // Check for WebGPU support
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        capabilities.webgpu = adapter !== null;
      } catch (e) {
        console.warn('WebGPU detection failed:', e);
      }
    }

    return capabilities;
  }

  /**
   * Load ONNX model
   * @param {string} modelPath - Path to .onnx model file
   * @param {function} progressCallback - Optional callback for loading progress
   * @returns {Promise<void>}
   */
  async loadModel(modelPath, progressCallback = null) {
    if (this.isLoaded) {
      console.log('Model already loaded');
      return;
    }

    if (this.isLoading) {
      console.warn('Model is already loading');
      return;
    }

    this.isLoading = true;
    this.modelPath = modelPath;

    try {
      // Detect capabilities
      const capabilities = await this.detectCapabilities();

      // Set execution provider preference
      // Note: WebGPU backend for ONNX Runtime requires separate bundle
      // For now, use WASM which is reliable and fast enough for 256x256 inference
      const sessionOptions = {
        executionProviders: ['wasm']
      };

      this.executionProvider = 'wasm';
      console.log('Using WASM execution provider for ONNX inference');

      // Optionally try WebGPU if explicitly available in ONNX Runtime
      // (requires onnxruntime-web/webgpu bundle)
      // if (capabilities.webgpu && window.ort?.env?.webgpu) {
      //   sessionOptions.executionProviders.unshift('webgpu');
      //   this.executionProvider = 'webgpu';
      //   console.log('Using WebGPU execution provider');
      // }

      // Report progress
      if (progressCallback) {
        progressCallback({ stage: 'downloading', progress: 0 });
      }

      // Load model
      const ort = window.ort;
      if (!ort) {
        throw new Error('ONNX Runtime not loaded. Include onnxruntime-web script.');
      }

      // Fetch model with progress tracking
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
      }

      const contentLength = response.headers.get('content-length');
      const total = parseInt(contentLength, 10);
      let loaded = 0;

      const reader = response.body.getReader();
      const chunks = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        loaded += value.length;

        if (progressCallback && total) {
          progressCallback({
            stage: 'downloading',
            progress: loaded / total
          });
        }
      }

      // Combine chunks
      const modelData = new Uint8Array(loaded);
      let position = 0;
      for (const chunk of chunks) {
        modelData.set(chunk, position);
        position += chunk.length;
      }

      if (progressCallback) {
        progressCallback({ stage: 'initializing', progress: 1.0 });
      }

      // Create inference session
      this.session = await ort.InferenceSession.create(
        modelData.buffer,
        sessionOptions
      );

      this.isLoaded = true;
      this.isLoading = false;

      console.log('Model loaded successfully');
      console.log('Input names:', this.session.inputNames);
      console.log('Output names:', this.session.outputNames);

      if (progressCallback) {
        progressCallback({ stage: 'ready', progress: 1.0 });
      }

    } catch (error) {
      this.isLoading = false;
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  /**
   * Run inference on input tensor
   * @param {Object} feeds - Input tensors { inputName: tensor }
   * @returns {Promise<Object>} Output tensors
   */
  async runInference(feeds) {
    if (!this.isLoaded || !this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    try {
      const results = await this.session.run(feeds);
      return results;
    } catch (error) {
      console.error('Inference failed:', error);
      throw error;
    }
  }

  /**
   * Get model info
   */
  getModelInfo() {
    if (!this.session) {
      return null;
    }

    return {
      inputNames: this.session.inputNames,
      outputNames: this.session.outputNames,
      executionProvider: this.executionProvider
    };
  }

  /**
   * Dispose model and free resources
   */
  dispose() {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.isLoaded = false;
    this.isLoading = false;
  }
}

/**
 * Portrait Segmentation Model
 * Extends MLInference with segmentation-specific preprocessing and postprocessing
 */
class PortraitSegmentation extends MLInference {
  constructor() {
    super();
    this.modelWidth = 256;
    this.modelHeight = 256;
    this.canvas = null;
    this.ctx = null;
  }

  /**
   * Preprocess video frame for segmentation model
   * Resize to 256x256 and normalize to [0, 1]
   * MediaPipe models expect NHWC format (batch, height, width, channels)
   * @param {HTMLVideoElement|HTMLCanvasElement|ImageBitmap} input - Input frame
   * @returns {Float32Array} Preprocessed tensor data in HWC format
   */
  preprocessFrame(input) {
    // Create canvas if needed
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = this.modelWidth;
      this.canvas.height = this.modelHeight;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }

    // Draw and resize input to model size
    this.ctx.drawImage(input, 0, 0, this.modelWidth, this.modelHeight);

    // Get image data
    const imageData = this.ctx.getImageData(0, 0, this.modelWidth, this.modelHeight);
    const { data } = imageData;

    // Convert to HWC format (height, width, channels) and normalize to [0, 1]
    // MediaPipe expects RGB values in range [0, 1]
    const inputSize = this.modelWidth * this.modelHeight;
    const tensorData = new Float32Array(inputSize * 3);

    for (let i = 0; i < inputSize; i++) {
      const pixelIndex = i * 4;  // RGBA format from canvas
      const tensorIndex = i * 3;  // RGB format for tensor (HWC)

      // Normalize RGB values from [0, 255] to [0, 1]
      tensorData[tensorIndex] = data[pixelIndex] / 255.0;         // R
      tensorData[tensorIndex + 1] = data[pixelIndex + 1] / 255.0; // G
      tensorData[tensorIndex + 2] = data[pixelIndex + 2] / 255.0; // B
    }

    return tensorData;
  }

  /**
   * Run segmentation inference on video frame
   * @param {HTMLVideoElement|HTMLCanvasElement|ImageBitmap} frame - Input frame
   * @returns {Promise<Float32Array>} Segmentation mask [0, 1]
   */
  async segmentFrame(frame) {
    if (!this.isLoaded) {
      throw new Error('Model not loaded');
    }

    // Preprocess
    const inputData = this.preprocessFrame(frame);

    // Create tensor in NHWC format (batch, height, width, channels)
    const ort = window.ort;
    const inputTensor = new ort.Tensor(
      'float32',
      inputData,
      [1, this.modelHeight, this.modelWidth, 3]
    );

    // Run inference
    const feeds = {};
    feeds[this.session.inputNames[0]] = inputTensor;
    const results = await this.runInference(feeds);

    // Get output tensor
    const outputName = this.session.outputNames[0];
    const outputTensor = results[outputName];

    return outputTensor.data;
  }

  /**
   * Postprocess segmentation mask
   * Apply threshold and optionally resize
   * @param {Float32Array} maskData - Raw model output
   * @param {number} threshold - Threshold value (default 0.5)
   * @returns {Uint8Array} Binary mask [0, 255]
   */
  postprocessMask(maskData, threshold = 0.5) {
    const maskSize = this.modelWidth * this.modelHeight;
    const binaryMask = new Uint8Array(maskSize);

    for (let i = 0; i < maskSize; i++) {
      // Apply sigmoid if needed and threshold
      let value = maskData[i];

      // If model outputs logits, apply sigmoid
      if (value < 0 || value > 1) {
        value = 1 / (1 + Math.exp(-value));
      }

      binaryMask[i] = value > threshold ? 255 : 0;
    }

    return binaryMask;
  }

  /**
   * Create debug visualization of mask
   * @param {Uint8Array} mask - Binary mask
   * @param {number} width - Display width
   * @param {number} height - Display height
   * @returns {ImageData} Visualization as ImageData
   */
  visualizeMask(mask, width = this.modelWidth, height = this.modelHeight) {
    const imageData = new ImageData(width, height);
    const data = imageData.data;

    for (let i = 0; i < mask.length; i++) {
      const pixelIndex = i * 4;
      const value = mask[i];

      data[pixelIndex] = value;     // R
      data[pixelIndex + 1] = value; // G
      data[pixelIndex + 2] = value; // B
      data[pixelIndex + 3] = 255;   // A
    }

    return imageData;
  }

  /**
   * Upsample mask to target dimensions using bilinear interpolation
   * @param {Uint8Array} mask - Input mask
   * @param {number} targetWidth - Target width
   * @param {number} targetHeight - Target height
   * @returns {Uint8Array} Upsampled mask
   */
  upsampleMask(mask, targetWidth, targetHeight) {
    const upsampled = new Uint8Array(targetWidth * targetHeight);

    const xRatio = this.modelWidth / targetWidth;
    const yRatio = this.modelHeight / targetHeight;

    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const srcX = x * xRatio;
        const srcY = y * yRatio;

        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, this.modelWidth - 1);
        const y2 = Math.min(y1 + 1, this.modelHeight - 1);

        const dx = srcX - x1;
        const dy = srcY - y1;

        // Bilinear interpolation
        const v11 = mask[y1 * this.modelWidth + x1];
        const v12 = mask[y1 * this.modelWidth + x2];
        const v21 = mask[y2 * this.modelWidth + x1];
        const v22 = mask[y2 * this.modelWidth + x2];

        const value = (1 - dx) * (1 - dy) * v11 +
                      dx * (1 - dy) * v12 +
                      (1 - dx) * dy * v21 +
                      dx * dy * v22;

        upsampled[y * targetWidth + x] = Math.round(value);
      }
    }

    return upsampled;
  }
}

// Export for use in other modules (ES6 modules for browser, CommonJS for Node.js)
export { MLInference, PortraitSegmentation };
