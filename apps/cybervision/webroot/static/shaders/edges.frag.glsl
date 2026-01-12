#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_algorithm;
uniform float u_threshold;
uniform float u_showOverlay;
uniform float u_invert;
uniform vec3 u_edgeColor;
uniform float u_thickness;

// Convert RGB to grayscale using luminance formula
float toGrayscale(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Sample pixel with bounds checking
float samplePixel(vec2 uv, vec2 offset) {
  vec2 sampleUV = uv + offset / u_resolution;
  if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) {
    return 0.0;
  }
  vec3 color = texture(u_video, sampleUV).rgb;
  return toGrayscale(color);
}

// Sobel edge detection
float sobelEdge(vec2 uv) {
  // Sobel kernels
  // Gx:             Gy:
  // [-1  0  1]      [-1 -2 -1]
  // [-2  0  2]      [ 0  0  0]
  // [-1  0  1]      [ 1  2  1]

  float tl = samplePixel(uv, vec2(-1.0, -1.0));
  float tc = samplePixel(uv, vec2(0.0, -1.0));
  float tr = samplePixel(uv, vec2(1.0, -1.0));
  float ml = samplePixel(uv, vec2(-1.0, 0.0));
  float mr = samplePixel(uv, vec2(1.0, 0.0));
  float bl = samplePixel(uv, vec2(-1.0, 1.0));
  float bc = samplePixel(uv, vec2(0.0, 1.0));
  float br = samplePixel(uv, vec2(1.0, 1.0));

  float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  return sqrt(gx * gx + gy * gy);
}

// Prewitt edge detection
float prewittEdge(vec2 uv) {
  // Prewitt kernels (uniform weights)
  // Gx:             Gy:
  // [-1  0  1]      [-1 -1 -1]
  // [-1  0  1]      [ 0  0  0]
  // [-1  0  1]      [ 1  1  1]

  float tl = samplePixel(uv, vec2(-1.0, -1.0));
  float tc = samplePixel(uv, vec2(0.0, -1.0));
  float tr = samplePixel(uv, vec2(1.0, -1.0));
  float ml = samplePixel(uv, vec2(-1.0, 0.0));
  float mr = samplePixel(uv, vec2(1.0, 0.0));
  float bl = samplePixel(uv, vec2(-1.0, 1.0));
  float bc = samplePixel(uv, vec2(0.0, 1.0));
  float br = samplePixel(uv, vec2(1.0, 1.0));

  float gx = -tl - ml - bl + tr + mr + br;
  float gy = -tl - tc - tr + bl + bc + br;

  return sqrt(gx * gx + gy * gy);
}

// Laplacian edge detection
float laplacianEdge(vec2 uv) {
  // Laplacian kernel (detects edges in all directions)
  // [ 0 -1  0]
  // [-1  4 -1]
  // [ 0 -1  0]

  float center = samplePixel(uv, vec2(0.0, 0.0));
  float top = samplePixel(uv, vec2(0.0, -1.0));
  float bottom = samplePixel(uv, vec2(0.0, 1.0));
  float left = samplePixel(uv, vec2(-1.0, 0.0));
  float right = samplePixel(uv, vec2(1.0, 0.0));

  float laplacian = 4.0 * center - top - bottom - left - right;
  return abs(laplacian);
}

// Canny-style edge detection (Sobel + non-maximum suppression)
float cannyEdge(vec2 uv) {
  // First, calculate Sobel gradients
  float tl = samplePixel(uv, vec2(-1.0, -1.0));
  float tc = samplePixel(uv, vec2(0.0, -1.0));
  float tr = samplePixel(uv, vec2(1.0, -1.0));
  float ml = samplePixel(uv, vec2(-1.0, 0.0));
  float mr = samplePixel(uv, vec2(1.0, 0.0));
  float bl = samplePixel(uv, vec2(-1.0, 1.0));
  float bc = samplePixel(uv, vec2(0.0, 1.0));
  float br = samplePixel(uv, vec2(1.0, 1.0));

  float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
  float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

  float magnitude = sqrt(gx * gx + gy * gy);

  // Simple non-maximum suppression approximation
  // Check perpendicular to gradient direction
  float angle = atan(gy, gx);
  float absAngle = abs(angle);

  float neighbor1;
  float neighbor2;

  // Quantize angle to 4 directions: 0, 45, 90, 135 degrees
  if (absAngle < 0.3927) { // ~22.5 degrees - horizontal
    neighbor1 = samplePixel(uv, vec2(-1.0, 0.0));
    neighbor2 = samplePixel(uv, vec2(1.0, 0.0));
  } else if (absAngle < 1.178) { // ~67.5 degrees - diagonal
    if (angle > 0.0) {
      neighbor1 = samplePixel(uv, vec2(-1.0, -1.0));
      neighbor2 = samplePixel(uv, vec2(1.0, 1.0));
    } else {
      neighbor1 = samplePixel(uv, vec2(-1.0, 1.0));
      neighbor2 = samplePixel(uv, vec2(1.0, -1.0));
    }
  } else if (absAngle < 1.963) { // ~112.5 degrees - vertical
    neighbor1 = samplePixel(uv, vec2(0.0, -1.0));
    neighbor2 = samplePixel(uv, vec2(0.0, 1.0));
  } else { // ~157.5 degrees - diagonal
    if (angle > 0.0) {
      neighbor1 = samplePixel(uv, vec2(-1.0, 1.0));
      neighbor2 = samplePixel(uv, vec2(1.0, -1.0));
    } else {
      neighbor1 = samplePixel(uv, vec2(-1.0, -1.0));
      neighbor2 = samplePixel(uv, vec2(1.0, 1.0));
    }
  }

  // Non-maximum suppression: suppress if not local maximum
  float center = samplePixel(uv, vec2(0.0, 0.0));
  if (center < neighbor1 || center < neighbor2) {
    return 0.0;
  }

  return magnitude;
}

// Apply edge thickness via dilation
float applyThickness(vec2 uv, float edgeValue, float thickness) {
  if (thickness <= 1.0) {
    return edgeValue;
  }

  float maxEdge = edgeValue;
  int radius = int(thickness);

  for (int dy = -3; dy <= 3; dy++) {
    for (int dx = -3; dx <= 3; dx++) {
      if (dx == 0 && dy == 0 || abs(dx) > radius || abs(dy) > radius) {
        continue;
      }

      vec2 offset = vec2(float(dx), float(dy));
      vec2 sampleUV = uv + offset / u_resolution;

      if (sampleUV.x >= 0.0 && sampleUV.x <= 1.0 && sampleUV.y >= 0.0 && sampleUV.y <= 1.0) {
        float neighborEdge;
        int algo = int(u_algorithm);

        if (algo == 0) {
          neighborEdge = sobelEdge(sampleUV);
        } else if (algo == 1) {
          neighborEdge = prewittEdge(sampleUV);
        } else if (algo == 2) {
          neighborEdge = laplacianEdge(sampleUV);
        } else {
          neighborEdge = cannyEdge(sampleUV);
        }

        maxEdge = max(maxEdge, neighborEdge);
      }
    }
  }

  return maxEdge;
}

void main() {
  // Run selected edge detection algorithm
  float edgeMagnitude;
  int algo = int(u_algorithm);

  if (algo == 0) {
    edgeMagnitude = sobelEdge(v_texCoord);
  } else if (algo == 1) {
    edgeMagnitude = prewittEdge(v_texCoord);
  } else if (algo == 2) {
    edgeMagnitude = laplacianEdge(v_texCoord);
  } else {
    edgeMagnitude = cannyEdge(v_texCoord);
  }

  // Apply thickness if needed
  if (u_thickness > 1.0) {
    edgeMagnitude = applyThickness(v_texCoord, edgeMagnitude, u_thickness);
  }

  // Apply threshold
  float isEdge = step(u_threshold, edgeMagnitude);

  // Determine output color based on overlay mode
  vec3 outputColor;

  if (u_showOverlay > 0.5) {
    // Overlay mode: show edges on original image
    vec3 originalColor = texture(u_video, v_texCoord).rgb;
    outputColor = mix(originalColor, u_edgeColor, isEdge);
  } else {
    // Black background mode
    vec3 backgroundColor = vec3(0.0);
    outputColor = mix(backgroundColor, u_edgeColor, isEdge);
  }

  // Apply invert if needed
  if (u_invert > 0.5) {
    outputColor = vec3(1.0) - outputColor;
  }

  fragColor = vec4(outputColor, 1.0);
}
