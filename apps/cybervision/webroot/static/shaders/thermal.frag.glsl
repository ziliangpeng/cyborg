#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;
uniform vec2 u_resolution;
uniform float u_palette;
uniform float u_contrast;
uniform float u_invert;

// Convert RGB to grayscale luminance
float luminance(vec3 color) {
  return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
}

// Apply contrast adjustment
float applyContrast(float lum, float contrast) {
  float centered = lum - 0.5;
  float adjusted = centered * contrast + 0.5;
  return clamp(adjusted, 0.0, 1.0);
}

// Classic Thermal: Black -> Blue -> Cyan -> Green -> Yellow -> Orange -> Red -> White
vec3 classicThermal(float t) {
  if (t < 0.143) {
    return mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), t / 0.143);
  } else if (t < 0.286) {
    return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.143) / 0.143);
  } else if (t < 0.429) {
    return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.286) / 0.143);
  } else if (t < 0.571) {
    return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.429) / 0.143);
  } else if (t < 0.714) {
    return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.571) / 0.143);
  } else if (t < 0.857) {
    return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.714) / 0.143);
  } else {
    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.857) / 0.143);
  }
}

// Infrared: Black -> Purple -> Blue -> Magenta -> Red -> Orange -> Yellow -> White
vec3 infraredPalette(float t) {
  if (t < 0.143) {
    return mix(vec3(0.0, 0.0, 0.0), vec3(0.3, 0.0, 0.5), t / 0.143);
  } else if (t < 0.286) {
    return mix(vec3(0.3, 0.0, 0.5), vec3(0.0, 0.0, 1.0), (t - 0.143) / 0.143);
  } else if (t < 0.429) {
    return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), (t - 0.286) / 0.143);
  } else if (t < 0.571) {
    return mix(vec3(1.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), (t - 0.429) / 0.143);
  } else if (t < 0.714) {
    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.571) / 0.143);
  } else if (t < 0.857) {
    return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.714) / 0.143);
  } else {
    return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.857) / 0.143);
  }
}

// Fire: Black -> Dark Red -> Red -> Orange -> Yellow -> White
vec3 firePalette(float t) {
  if (t < 0.2) {
    return mix(vec3(0.0, 0.0, 0.0), vec3(0.5, 0.0, 0.0), t / 0.2);
  } else if (t < 0.4) {
    return mix(vec3(0.5, 0.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.2) / 0.2);
  } else if (t < 0.6) {
    return mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.4) / 0.2);
  } else if (t < 0.8) {
    return mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.6) / 0.2);
  } else {
    return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.8) / 0.2);
  }
}

// Map luminance to thermal color based on palette
vec3 thermalColor(float lum, int palette) {
  if (palette == 0) {
    return classicThermal(lum);
  } else if (palette == 1) {
    return infraredPalette(lum);
  } else {
    return firePalette(lum);
  }
}

void main() {
  // Sample input color
  vec3 color = texture(u_video, v_texCoord).rgb;

  // Calculate luminance
  float lum = luminance(color);

  // Apply contrast
  lum = applyContrast(lum, u_contrast);

  // Apply inversion
  if (u_invert > 0.5) {
    lum = 1.0 - lum;
  }

  // Map to thermal color
  vec3 thermalCol = thermalColor(lum, int(u_palette));

  fragColor = vec4(thermalCol, 1.0);
}
