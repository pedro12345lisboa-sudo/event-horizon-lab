#version 330 core

// ===========================================================================
// Grade de espaço-tempo — vertex shader (GLSL 330, ModernGL)
//
// A grade é um reticulado de linhas (GL_LINES). Toda a "deformação
// gravitacional" acontece AQUI: a posição original `in_pos` é deslocada em
// direção às massas (rede 3D) ou para baixo no eixo z (lençol) e só então
// projetada na tela. Assim a geometria da grade nunca precisa ser reescrita
// no CPU — as massas andam e o "tecido" acompanha.
//
// Com u_num_masses == 0 (Etapa 1) displace() devolve (0,0,0) e a grade
// aparece reta: é exatamente o caso estático.
//
// UNIDADES: geometria geométrica "de cena" (G = c = 1, comprimentos em
// unidades arbitrárias). Nada aqui está em escala real da Terra/Lua.
// ===========================================================================

layout(location = 0) in vec3 in_pos;      // posição da grade (antes da deformação)

uniform mat4  u_mvp;        // projeção * visão (matriz de clip)
uniform vec3  u_cam_pos;    // posição da câmera (fade por distância)

// --- massas ---------------------------------------------------------------
uniform int   u_num_masses;   // massas ativas (0 = grade estática)
uniform vec3  u_mass_pos[8];  // centro de cada massa
uniform float u_mass[8];      // massa (positiva = atrai)
uniform float u_k;            // constante de acoplamento da deformação
uniform float u_eps;          // suavização do núcleo (evita divisão por zero)
uniform float u_max_disp;     // clamp da magnitude do deslocamento

// --- modo -----------------------------------------------------------------
uniform int   u_mode;         // 0 = rede 3D, 1 = lençol (rubber sheet)
uniform int   u_funnel;       // 1 = funil exato de Schwarzschild (só lençol)
uniform float u_M;            // massa do buraco negro (M = 1)

out vec3  v_world;     // posição já deformada
out float v_camdist;   // distância à câmera (para o fade)
out float v_disp;      // magnitude do deslocamento (debug/visual)

// ---------------------------------------------------------------------------
// Rede 3D: desloca cada vértice EM DIREÇÃO ÀS MASSAS
//
//     d(x) = Σ_i  k · m_i · (x_i − x) / (|x − x_i|² + ε²)^(3/2)
//
// Vetor (x_i − x) aponta de x até a massa: o vértice é puxado para perto
// dela, então as linhas se concentram ao redor das massas (é o campo de
// força newtoniano −∇Φ com potencial suavizado Φ = −k·m/√(r²+ε²)).
// O ε impede a singularidade em x = x_i; o clamp em u_max_disp impede que
// vértices fiquem tão puxados que as linhas se atravessem de forma feia.
// ---------------------------------------------------------------------------
vec3 displace_rede(vec3 x) {
    vec3 d = vec3(0.0);
    for (int i = 0; i < 8; ++i) {
        if (i >= u_num_masses) break;
        vec3 r = u_mass_pos[i] - x;              // de x até a massa
        float r2 = dot(r, r) + u_eps * u_eps;    // (|x − x_i|² + ε²)
        d += (u_k * u_mass[i] / (r2 * sqrt(r2))) * r;   // / (r²)^(3/2)
    }
    float len = length(d);
    if (len > u_max_disp) {
        d *= u_max_disp / len;                   // clamp do deslocamento
    }
    return d;
}

// ---------------------------------------------------------------------------
// Lençol: só o eixo z é afetado (a grade 2D vive em z = 0 antes da deformação)
//
//   · massas:  z(x,y) = −Σ_i k·m_i / √(|r − r_i|² + ε²)   (poço potencial)
//              o sinal negativo faz o tecido AFUNDAR sob a massa;
//   · funil (u_funnel == 1): substitui a soma pela imersão exata de Flamm
//              para um horizonte de Schwarzschild, M = 1:
//                  z_imersao(r) = −2·√(2M(r − 2M))   (só para r ≥ 2M)
//              Essa é a fórmula do projeto, escrita no eixo de imersão, que
//              aponta do poço para fora. Na cena o eixo z para cima e o funil
//              tem que AFUNDAR até o horizonte, então a altura desenhada é o
//              oposto:
//                  z_cena(r) = −z_imersao(r) = +2·√(2M(r − 2M)),
//              que é zero no horizonte e cresce com r (paraboloid de Flamm).
// ---------------------------------------------------------------------------
float lencol_z(vec2 p) {
    if (u_funnel == 1) {
        float rh = 2.0 * u_M;                 // horizonte de Schwarzschild
        float r = max(length(p), rh);         // fórmula só vale para r ≥ 2M
        // altura do funil ( = −z_imersao ): zero em r = 2M, afunda para fora
        return 2.0 * sqrt(2.0 * u_M * (r - rh));
    }

    float z = 0.0;
    for (int i = 0; i < 8; ++i) {
        if (i >= u_num_masses) break;
        vec2 dp = u_mass_pos[i].xy - p;
        float d = sqrt(dot(dp, dp) + u_eps * u_eps);
        z -= u_k * u_mass[i] / d;
    }
    if (z < -u_max_disp) z = -u_max_disp;     // o poço não afunda além disso
    return z;
}

void main() {
    vec3 disp = (u_mode == 0)
        ? displace_rede(in_pos)
        : vec3(0.0, 0.0, lencol_z(in_pos.xy));

    vec3 w = in_pos + disp;
    v_world   = w;
    v_disp    = length(disp);
    v_camdist = distance(u_cam_pos, w);
    gl_Position = u_mvp * vec4(w, 1.0);
}
