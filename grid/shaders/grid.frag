#version 330 core

// ===========================================================================
// Grade de espaço-tempo — fragment shader
//
// Linhas amarelo-esverdeadas finas com fade por distância da câmera. O
// blending é ADITIVO (SRC_ALPHA, ONE) configurado no Python: a cor de cada
// fragmento é somada ao que já estava no framebuffer, então onde duas linhas
// se cruzam (ou onde a grade é mais densa, perto das massas) o brilho
// acumula — é o "glow" do tecido do espaço-tempo.
// ===========================================================================

in vec3  v_world;
in float v_camdist;
in float v_disp;

uniform vec3  u_color;      // cor base das linhas (amarelo-esverdeado)
uniform float u_alpha;      // brilho base de UMA linha
uniform float u_fade_near;  // até essa distância a linha está 100% visível
uniform float u_fade_far;   // além dessa distância a linha desaparece

out vec4 fragColor;

void main() {
    // smoothstep: 1 até u_fade_near, cai suavemente até 0 em u_fade_far.
    // Isso esconde o corte abrupto do reticulado e dá sensação de profundidade.
    float fade = 1.0 - smoothstep(u_fade_near, u_fade_far, v_camdist);

    // Com blend (SRC_ALPHA, ONE): o que entra no framebuffer é
    // u_color * (u_alpha * fade) somado ao destino.
    fragColor = vec4(u_color, u_alpha * fade);
}
