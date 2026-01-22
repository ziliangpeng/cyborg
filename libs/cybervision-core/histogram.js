/**
 * Real-time RGB Histogram calculation and visualization.
 */
export class Histogram {
  constructor(targetCanvas) {
    this.targetCanvas = targetCanvas;
    this.ctx = targetCanvas.getContext('2d');
    
    // Offscreen canvas for sampling
    this.sampleCanvas = document.createElement('canvas');
    this.sampleCtx = this.sampleCanvas.getContext('2d', { willReadFrequently: true });
    
    // Sampling resolution (higher = more accurate but slower)
    this.sampleWidth = 160;
    this.sampleHeight = 120;
    this.sampleCanvas.width = this.sampleWidth;
    this.sampleCanvas.height = this.sampleHeight;

    // Buffers for histogram data
    this.r = new Uint32Array(256);
    this.g = new Uint32Array(256);
    this.b = new Uint32Array(256);
    this.l = new Uint32Array(256); // Luminance
  }

  /**
   * Update and draw the histogram.
   * @param {HTMLVideoElement|HTMLCanvasElement} source - The source image
   */
  update(source) {
    if (!source || (source.videoWidth === 0 && source.width === 0)) return;

    // 1. Sample the source image to the small canvas
    this.sampleCtx.drawImage(source, 0, 0, this.sampleWidth, this.sampleHeight);
    
    // 2. Get pixel data
    const imageData = this.sampleCtx.getImageData(0, 0, this.sampleWidth, this.sampleHeight);
    const data = imageData.data;
    
    // 3. Reset histogram buffers
    this.r.fill(0);
    this.g.fill(0);
    this.b.fill(0);
    this.l.fill(0);

    let maxVal = 0;

    // 4. Calculate frequencies
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      this.r[r]++;
      this.g[g]++;
      this.b[b]++;
      
      // Standard luminance calculation
      const l = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      this.l[l]++;

      // Keep track of max for scaling
      maxVal = Math.max(maxVal, this.r[r], this.g[g], this.b[b], this.l[l]);
    }

    // 5. Draw the histogram
    this.draw(maxVal);
  }

  draw(maxVal) {
    const { width, height } = this.targetCanvas;
    const ctx = this.ctx;

    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, width, height);

    if (maxVal === 0) return;

    // Draw channels
    this.drawChannel(this.r, 'rgba(255, 0, 0, 0.6)', maxVal);
    this.drawChannel(this.g, 'rgba(0, 255, 0, 0.6)', maxVal);
    this.drawChannel(this.b, 'rgba(0, 0, 255, 0.6)', maxVal);
    this.drawChannel(this.l, 'rgba(255, 255, 255, 0.4)', maxVal);
    
    // Draw border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.strokeRect(0, 0, width, height);
  }

  drawChannel(data, color, maxVal) {
    const { width, height } = this.targetCanvas;
    const ctx = this.ctx;
    
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    
    const scaleX = width / 255;
    const scaleY = height / maxVal;

    ctx.moveTo(0, height - data[0] * scaleY);
    for (let i = 1; i < 256; i++) {
      ctx.lineTo(i * scaleX, height - data[i] * scaleY);
    }
    
    ctx.stroke();

    // Fill area
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    const fillColor = color.replace('0.6', '0.2').replace('0.4', '0.1');
    ctx.fillStyle = fillColor;
    ctx.fill();
  }
}
