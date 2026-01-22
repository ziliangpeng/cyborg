/**
 * Real-time Image Analysis Visualizations
 */
export class Histogram {
  constructor(canvases) {
    this.canvases = canvases; // { rgb: canvas, hue: canvas, sat: canvas }
    this.ctxs = {
      rgb: canvases.rgb?.getContext('2d'),
      hue: canvases.hue?.getContext('2d'),
      sat: canvases.sat?.getContext('2d')
    };
    
    // Offscreen canvas for sampling
    this.sampleCanvas = document.createElement('canvas');
    this.sampleCtx = this.sampleCanvas.getContext('2d', { willReadFrequently: true });
    
    // Sampling resolution
    this.sampleWidth = 160;
    this.sampleHeight = 120;
    this.sampleCanvas.width = this.sampleWidth;
    this.sampleCanvas.height = this.sampleHeight;

    // Buffers
    this.r = new Uint32Array(256);
    this.g = new Uint32Array(256);
    this.b = new Uint32Array(256);
    this.l = new Uint32Array(256); // Luminance
    this.hue = new Uint32Array(256); // 0-255 mapped from 0-360
    this.sat = new Uint32Array(256); // 0-255 mapped from 0-100%
  }

  update(source) {
    if (!source || (source.videoWidth === 0 && source.width === 0)) return;

    this.sampleCtx.drawImage(source, 0, 0, this.sampleWidth, this.sampleHeight);
    const imageData = this.sampleCtx.getImageData(0, 0, this.sampleWidth, this.sampleHeight);
    const data = imageData.data;
    
    this.r.fill(0);
    this.g.fill(0);
    this.b.fill(0);
    this.l.fill(0);
    this.hue.fill(0);
    this.sat.fill(0);

    let maxRGB = 0;
    let maxHue = 0;
    let maxSat = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // RGB + Luma
      this.r[r]++;
      this.g[g]++;
      this.b[b]++;
      const l = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      this.l[l]++;
      maxRGB = Math.max(maxRGB, this.r[r], this.g[g], this.b[b], this.l[l]);

      // HSL Conversion (Simplified for speed)
      const rf = r / 255, gf = g / 255, bf = b / 255;
      const max = Math.max(rf, gf, bf), min = Math.min(rf, gf, bf);
      const d = max - min;
      
      // Saturation
      const sVal = max === 0 ? 0 : d / max;
      const sIdx = Math.round(sVal * 255);
      this.sat[sIdx]++;
      maxSat = Math.max(maxSat, this.sat[sIdx]);

      // Hue
      if (d !== 0) {
        let h;
        if (max === rf) h = (gf - bf) / d + (gf < bf ? 6 : 0);
        else if (max === gf) h = (bf - rf) / d + 2;
        else h = (rf - gf) / d + 4;
        const hIdx = Math.round((h / 6) * 255);
        this.hue[hIdx]++;
        maxHue = Math.max(maxHue, this.hue[hIdx]);
      }
    }

    if (this.ctxs.rgb) this.drawRGB(maxRGB);
    if (this.ctxs.hue) this.drawHue(maxHue);
    if (this.ctxs.sat) this.drawSat(maxSat);
  }

  drawRGB(maxVal) {
    const ctx = this.ctxs.rgb;
    const { width, height } = this.canvases.rgb;
    this.prepareCanvas(ctx, width, height);

    this.drawChannel(ctx, this.r, 'rgba(255, 60, 60, 0.6)', maxVal, width, height);
    this.drawChannel(ctx, this.g, 'rgba(60, 255, 60, 0.6)', maxVal, width, height);
    this.drawChannel(ctx, this.b, 'rgba(60, 160, 255, 0.6)', maxVal, width, height);
    this.drawChannel(ctx, this.l, 'rgba(255, 255, 255, 0.4)', maxVal, width, height);
  }

  drawHue(maxVal) {
    const ctx = this.ctxs.hue;
    const { width, height } = this.canvases.hue;
    this.prepareCanvas(ctx, width, height);

    const scaleX = width / 255;
    const scaleY = height / (maxVal || 1);

    for (let i = 0; i < 256; i++) {
      if (this.hue[i] > 0) {
        const h = Math.round((i / 255) * 360);
        ctx.fillStyle = `hsla(${h}, 80%, 60%, 0.8)`;
        const barHeight = this.hue[i] * scaleY;
        ctx.fillRect(i * scaleX, height - barHeight, Math.max(1, scaleX), barHeight);
      }
    }
  }

  drawSat(maxVal) {
    const ctx = this.ctxs.sat;
    const { width, height } = this.canvases.sat;
    this.prepareCanvas(ctx, width, height);
    this.drawChannel(ctx, this.sat, 'rgba(0, 220, 255, 0.7)', maxVal, width, height);
  }

  prepareCanvas(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.strokeRect(0, 0, width, height);
  }

  drawChannel(ctx, data, color, maxVal, width, height) {
    if (!maxVal) return;
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

    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.fillStyle = color.replace('0.7', '0.1').replace('0.6', '0.1').replace('0.4', '0.05');
    ctx.fill();
  }
}