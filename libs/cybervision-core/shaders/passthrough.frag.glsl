#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_video;

void main() {
  fragColor = texture(u_video, v_texCoord);
}
