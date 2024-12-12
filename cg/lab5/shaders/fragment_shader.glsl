#version 330 core

in vec3 fragPos;
in vec3 fragNormal;

out vec4 fragColor;

uniform vec3 objectColor;
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 cameraPos;
uniform float refractiveIndex;

void main()
{
    // Compute refraction
    vec3 I = normalize(fragPos - cameraPos);
    vec3 N = normalize(fragNormal);
    float eta = 1.0 / refractiveIndex;
    vec3 refractDir = refract(I, N, eta);

    // Compute lighting
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(lightDir, N), 0.0);

    vec3 ambient = 0.1 * objectColor;
    vec3 diffuse = diff * lightColor * objectColor;
    vec3 refracted = 0.9 * refractDir; // Adjust for visual clarity

    fragColor = vec4(ambient + diffuse + refracted, 1.0);
}
