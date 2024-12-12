#version 330 core

uniform sampler2D shadowMap;

out vec4 FragColor;

void main()
{
    vec2 texCoords = gl_FragCoord.xy / textureSize(shadowMap, 0);
    float depth = texture(shadowMap, texCoords).r;
    FragColor = vec4(vec3(depth), 1.0);
}
