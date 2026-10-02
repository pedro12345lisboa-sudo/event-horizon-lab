// ============================================================
// geodesics.cpp — Buraco Negro de Schwarzschild (G = c = M = 1)
// ============================================================
// Integra geodésicas de fótons (nulas) e de partícula (tipo-tempo)
// usando RK4, e escreve CSVs para visualização em Python.
//
// Física:
//   Métrica de Schwarzschild: ds² = -(1-2M/r)dt² + (1-2M/r)⁻¹dr² + r²dΩ²
//   Fóton:  d²u/dφ² + u = 3Mu²   (u = 1/r, plano equatorial)
//   Partícula em queda radial:  d²r/dτ² = -M/r²
//   Coordenadas de Eddington-Finkelstein: v = t + ∫dr/(1-2M/r)
//     = t + r + 2M ln|r/(2M) - 1|   (atravessa o horizonte sem divergir)
// ============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <array>

// --- Constantes físicas (unidades naturais) ---
static constexpr double M = 1.0;       // massa do buraco negro
static constexpr double r_s = 2.0 * M; // raio de Schwarzschild = 2
static constexpr double b_crit_exact = 3.0 * std::sqrt(3.0) * M; // ≈ 5.19615

// ============================================================
// Integrador RK4 genérico para sistemas de ODE de primeira ordem
// ============================================================
// y' = f(y),  y ∈ ℝᴺ.  Passo h (pode ser negativo).
template <size_t N, typename F>
std::array<double, N> rk4(const std::array<double, N>& y, double h, F&& f) {
    auto k1 = f(y);
    std::array<double, N> y2{};
    for (size_t i = 0; i < N; ++i) y2[i] = y[i] + 0.5 * h * k1[i];
    auto k2 = f(y2);
    std::array<double, N> y3{};
    for (size_t i = 0; i < N; ++i) y3[i] = y[i] + 0.5 * h * k2[i];
    auto k3 = f(y3);
    std::array<double, N> y4{};
    for (size_t i = 0; i < N; ++i) y4[i] = y[i] + h * k3[i];
    auto k4 = f(y4);
    std::array<double, N> out{};
    for (size_t i = 0; i < N; ++i)
        out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    return out;
}

// ============================================================
// Geodésica de fóton — equação de órbita em u = 1/r
// ============================================================
// A equação geodésica nula no plano equatorial é:
//   d²u/dφ² + u = 3Mu²
// Condição de energia:  (du/dψ)² = 1/b² - u²(1 - 2Mu)
// onde b = L/E é o parâmetro de impacto e ψ = π - φ (ψ aumenta
// à medida que o fóton se aproxima vindo da esquerda).
//
// Tempo coordenado ao longo da geodésica (para sincronizar a animação
// com a partícula no relógio de Schwarzschild):
//   dt/dφ = (1/b) · r² / (1 - 2M/r)   e como φ = π - ψ ⟹ dφ = -dψ
//   ⟹ dt/dψ = 1 / (b u² (1 - 2Mu))
// Congela (dt/dψ = 0) dentro do horizonte, onde t é espaço-tempo e o
// tratamento é o mesmo usado na linha de mundo da partícula.
//
// Estrutura do estado: y = (u, u', ψ, t)
struct PhotonResult {
    bool escaped;       // true = escapou, false = capturado
    std::vector<double> x, y;  // coordenadas cartesianas (plano z=0)
    std::vector<double> t;     // tempo coordenado de Schwarzschild
};

// Função RHS para fóton: y = (u, u', ψ, t)
// u'' = 3Mu² - u,  ψ' = 1,  t' = dt/dψ
auto photon_rhs(double b) {
    return [b](const std::array<double, 4>& y) -> std::array<double, 4> {
        double u  = y[0];
        double om = 1.0 - 2.0 * M * u;                 // f = 1 - 2M/r
        double dtdpsi = (om > 1e-9 && u > 1e-14)
                      ? 1.0 / (b * u * u * om) : 0.0;  // 0 dentro do horizonte
        return { y[1], 3.0 * M * u * u - u, 1.0, dtdpsi };
    };
}

// Integra uma geodésica de fóton.
// Estratégia única para todos os b:
//   Começa longe (u = u_inf, ψ = 0) com u' = sqrt(1/b² - u_inf² + 2Mu_inf³) > 0
//   (fóton se aproximando). Integra com dψ > 0.
//   Se u' fica negativo depois de crescer → periápsio → u diminui → escapou.
//   Se u > u_max → cruzou o horizonte → capturado.
PhotonResult integrate_photon(double b, double dpsi = 0.002,
                              double u_inf = 1e-6, double u_max = 50.0) {
    PhotonResult res;
    res.escaped = false;

    // Condição inicial: fóton longe, vindo da esquerda, aproximando-se
    double u_prime0 = std::sqrt(std::max(0.0,
        1.0 / (b * b) - u_inf * u_inf + 2.0 * M * u_inf * u_inf * u_inf));
    std::array<double, 4> y{ u_inf, u_prime0, 0.0, 0.0 }; // t(ψ=0) = 0

    auto to_cart = [&](const std::array<double, 4>& yy) {
        double u = yy[0];
        if (u < 1e-14) u = 1e-14;
        double r = 1.0 / u;
        double phi = M_PI - yy[2]; // ψ = π - φ
        res.x.push_back(r * std::cos(phi));
        res.y.push_back(r * std::sin(phi));
        res.t.push_back(yy[3]);    // tempo coordenado de Schwarzschild
    };

    to_cart(y);

    bool past_periapsis = false;
    for (int i = 0; i < 400000; ++i) {
        // Passo adaptativo: dt/dψ = 1/(b u² f) é gigante no campo distante
        // (u → 0) e varia muitas ordens de grandeza dentro de um passo fixo.
        // Limita Δu ≤ 0.05·u (variação relativa ≤ 5% por passo) para que o
        // RK4 integre t com precisão. Longe isso custa ~170 passos log;
        // perto do buraco negro vira o passo fixo dpsi de novo.
        double h = dpsi;
        double up = std::abs(y[1]);
        if (up > 1e-300 && y[0] > 0.0)
            h = std::min(dpsi, 0.05 * y[0] / up);
        y = rk4<4>(y, h, photon_rhs(b));

        if (y[1] < 0.0) past_periapsis = true;         // passou do periápsio (u' < 0)

        // ESCAPE: depois do periápsio, u cai de volta até u_inf (r → ∞)
        // (u pode ficar negativo numericamente quando r → ∞; trata como escape)
        if (past_periapsis && y[0] <= u_inf) {
            res.escaped = true;
            if (y[0] > 0.0) to_cart(y);
            break;
        }
        // CAPTURA: u ultrapassa u_max → cruzou o horizonte (r < 1/u_max)
        if (y[0] > u_max) break;

        if (y[0] > 0.0) to_cart(y);
    }
    return res;
}

// ============================================================
// Geodésica de partícula — queda radial a partir do repouso
// ============================================================
// EDOs (τ como parâmetro):
//   dr/dτ  = vr
//   dvr/dτ = -M/r²              (queda radial em Schwarzschild)
//   dv/dτ  = M / (ε r²)         ← velocidade local v = |dr/dτ|/ε (regular sempre)
//   dt/dτ  = ε / (1 - 2M/r)     ← diverge no horizonte (r → 2M)
//   dvEF/dτ = 1 / (ε + √(ε² - (1-2M/r)))   ← tempo de Eddington-Finkelstein,
//             vEF = t + r + 2M ln|r/2M - 1|, forma COMOVEL sem polo: o polo
//             de dt/dτ cancela exatamente com o de dr/dτ + 2M/(r-2M)·dr/dτ
// onde ε = sqrt(1 - 2M/r₀) é a energia conservada.
//
// t de Schwarzschild só é integrado fora do horizonte (f = 1-2M/r > 0);
// dentro do horizonte ele é congelado (não é definido para um observador
// caindo — a cruzagem acontece em t = ∞ exato, ver painel espaço-tempo).
struct ParticleResult {
    std::vector<double> tau, r, v, t, vef; // trajetória da linha de mundo
};

ParticleResult integrate_particle(double r0, double dtau = 0.002,
                                  double r_end = 0.02) {
    ParticleResult res;
    double eps2 = 1.0 - 2.0 * M / r0;  // ε² (energia ao quadrado)
    double eps  = std::sqrt(eps2);

    // Estado: y = (r, vr, v_local, t, v_EF)
    auto rhs = [&](const std::array<double, 5>& y) -> std::array<double, 5> {
        double r  = y[0];
        double vr = y[1];
        // v = velocidade local medida por observadores estáticos:
        //   v = |dr/dτ|/ε = √(1 - (1-2M/r)/ε²)   (v=0 no repouso, v=1 no horizonte)
        // Derivando: dv/dτ = M/(ε r²) — regular no repouso E no horizonte
        // (a equação anterior v' = (1-2M/r)/(ε²-(1-2M/r)) era singular no
        //  repouso inicial, onde ε² = 1-2M/r₀ exatamente, dando v ~ 1e11).
        double dvdt = M / (eps * r * r);
        double f = 1.0 - 2.0 * M / r;
        // dt/dτ = ε/f: integrado só fora (f > 0); dentro congela (t indefinido)
        double dtdt = (f > 0.0) ? eps / f : 0.0;
        // dvEF/dτ = 1/(ε + √(ε²-f))  — denominador ≥ ε > 0, nunca diverge
        double dvef = 1.0 / (eps + std::sqrt(eps2 - f));
        return { vr, -M / (r * r), dvdt, dtdt, dvef };
    };

    // vEF(0) = t₀ + r₀ + 2 ln|r₀/2 - 1| (condição inicial do tempo EF)
    double vef0 = r0 + 2.0 * std::log(std::abs(r0 / (2.0 * M) - 1.0));
    std::array<double, 5> y{ r0, 0.0, 0.0, 0.0, vef0 }; // parte do repouso em r₀

    // Grava o ponto inicial
    res.tau.push_back(0.0);
    res.r.push_back(r0);
    res.v.push_back(0.0);
    res.t.push_back(0.0);
    res.vef.push_back(vef0);

    double tau = 0.0;
    for (int i = 0; i < 200000; ++i) {
        // Passo adaptativo: menor quando r está perto do horizonte
        double h = dtau;
        if (y[0] < r_s + 0.5) h = dtau * 0.25;
        if (y[0] < r_s + 0.05) h = dtau * 0.05;

        y = rk4<5>(y, h, rhs);
        tau += h;

        // Trunca se r cai abaixo do limite
        if (y[0] < r_end) {
            y[0] = r_end;
            res.tau.push_back(tau);
            res.r.push_back(r_end);
            res.v.push_back(y[2]);
            res.t.push_back(y[3]);
            res.vef.push_back(y[4]);
            break;
        }

        res.tau.push_back(tau);
        res.r.push_back(y[0]);
        res.v.push_back(y[2]);
        res.t.push_back(y[3]);
        res.vef.push_back(y[4]);

        if (y[1] > 0.0 && y[0] < r0 * 0.5) break; // subindo de volta? (não deve acontecer)
    }
    return res;
}

// ============================================================
// Escrita de CSVs
// ============================================================
void write_photon_csv(int id, const PhotonResult& ph, const std::string& outdir) {
    std::ostringstream fname;
    fname << outdir << "/ray_" << std::setw(2) << std::setfill('0') << id << ".csv";
    std::ofstream f(fname.str());
    f << std::setprecision(17); // 6 dígitos (padrão) destruiriam os incrementos de t
    f << "x,y,z,t\n";
    for (size_t i = 0; i < ph.x.size(); ++i)
        f << ph.x[i] << "," << ph.y[i] << ",0," << ph.t[i] << "\n";
}

void write_particle_csv(const ParticleResult& p, const std::string& outdir) {
    std::ofstream f(outdir + "/particle_worldline.csv");
    f << std::setprecision(17);
    f << "tau,r,v,t,v_ef,dtau_dt,redshift\n";
    for (size_t i = 0; i < p.tau.size(); ++i) {
        double ri = p.r[i];
        double f_dtau = (ri > r_s) ? std::sqrt(1.0 - 2.0 * M / ri) : 0.0;
        double z = (ri > r_s) ? 1.0 / f_dtau - 1.0 : 1e12;
        // Dentro do horizonte não existem observadores estáticos →
        // a velocidade local não é definida; reporta o limite v = 1.
        double v = (ri > r_s) ? p.v[i] : 1.0;
        f << p.tau[i] << "," << ri << "," << v << "," << p.t[i]
          << "," << p.vef[i] << "," << f_dtau << "," << z << "\n";
    }
}

// ============================================================
// Testes numéricos
// ============================================================

// Teste 1: parâmetro de impacto crítico via bissecção
double find_bcrit() {
    auto is_captured = [](double b) -> bool {
        PhotonResult ph = integrate_photon(b, 0.004);
        return !ph.escaped;
    };
    double lo = 3.0, hi = 8.0; // intervalo que contém 3√3 ≈ 5.196
    for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (lo + hi);
        if (is_captured(mid)) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// Teste 2: deflexão em campo fraco  α ≈ 4M/b  para b = 100
double measure_deflection(double b) {
    // Integra do longe (ψ₀ = 0, u = u_inf) até o fóton escapar de volta.
    // α = ψ_final - ψ₀ - π   (linha reta daria Δψ = π)
    auto rhs = photon_rhs(b);
    double dpsi = 0.0001;    // passo menor para precisão < 2%
    double u_inf = 1e-8;     // começa mais longe (r = 10⁸)
    double u_prime0 = std::sqrt(std::max(0.0,
        1.0 / (b * b) - u_inf * u_inf + 2.0 * M * u_inf * u_inf * u_inf));
    std::array<double, 4> y{ u_inf, u_prime0, 0.0, 0.0 };

    bool past_periapsis = false;
    double psi_final = 0.0;
    for (int i = 0; i < 10000000; ++i) {
        y = rk4<4>(y, dpsi, rhs);
        if (y[1] < 0.0) past_periapsis = true;
        if (past_periapsis && y[0] <= u_inf) {
            psi_final = y[2];
            break;
        }
        if (y[0] > 50.0) { psi_final = y[2]; break; } // capturado (não deveria com b=100)
    }
    // Δψ = ψ_final - 0,  α = Δψ - π
    return psi_final - M_PI;
}

// Teste 3: tempo próprio da queda radial
double measure_proper_time(double r0) {
    ParticleResult p = integrate_particle(r0, 0.001, 0.005);
    return p.tau.back();
}

// Teste 4: conservação de energia e momento angular (fóton)
// Para fóton: ε_cons = (du/dψ)² + u² - 2Mu³ deve ser constante = 1/b²
bool check_photon_conservation(double b, double& max_err) {
    auto rhs = photon_rhs(b);
    double dpsi = 0.001;
    double u_inf = 1e-6;
    double expected = 1.0 / (b * b);

    double u_prime0 = std::sqrt(std::max(0.0,
        expected - u_inf * u_inf + 2.0 * M * u_inf * u_inf * u_inf));
    std::array<double, 4> y{ u_inf, u_prime0, 0.0, 0.0 };

    max_err = 0.0;
    bool past_periapsis = false;
    for (int i = 0; i < 100000; ++i) {
        double cons = y[1] * y[1] + y[0] * y[0] - 2.0 * M * y[0] * y[0] * y[0];
        double err = std::abs(cons - expected);
        if (err > max_err) max_err = err;
        y = rk4<4>(y, dpsi, rhs);
        if (y[1] < 0.0) past_periapsis = true;
        if (past_periapsis && y[0] <= u_inf) break;
        if (y[0] > 10.0 || y[0] < 1e-10) break;
    }
    return max_err < 1e-6;
}

// ============================================================
// Programa principal
// ============================================================
int main() {
    const std::string outdir = "blackhole/out";

    std::cout << "=== Buraco Negro de Schwarzschild — Geodésicas RK4 ===" << std::endl;
    std::cout << "Unidades: G = c = M = 1,  r_s = 2M = 2" << std::endl;
    std::cout << "Esfera de fótons: r = 3M = 3" << std::endl;
    std::cout << "ISCO: r = 6M = 6" << std::endl;
    std::cout << "b_crítico exato = 3√3 M = " << b_crit_exact << std::endl;
    std::cout << std::endl;

    // ---- Gera geodésicas de fóton para visualização ----
    // b < b_crit → capturado; b > b_crit → escapa
    std::vector<double> b_values = {
        2.0, 3.0, 4.0, 5.0,        // capturados (b < 5.196)
        5.19,                        // quase crítico (capturado)
        5.21,                        // quase crítico (escapa)
        6.0, 8.0, 10.0, 15.0, 20.0, // escapa
        -2.0, -4.0, -6.0, -10.0     // abaixo do eixo (simetria)
    };

    std::cout << "Gerando " << b_values.size() << " geodésicas de fóton..." << std::endl;
    for (size_t i = 0; i < b_values.size(); ++i) {
        PhotonResult ph = integrate_photon(std::abs(b_values[i]));
        // Reflete em y para valores negativos de b
        if (b_values[i] < 0.0) {
            for (auto& yi : ph.y) yi = -yi;
        }
        write_photon_csv(static_cast<int>(i), ph, outdir);
        std::cout << "  ray_" << std::setw(2) << std::setfill('0') << i
                  << "  b = " << std::setw(6) << b_values[i]
                  << "  → " << (ph.escaped ? "ESCAPA" : "CAPTURADO")
                  << "  (" << ph.x.size() << " pontos)" << std::endl;
    }

    // ---- Gera linha de mundo da partícula ----
    std::cout << std::endl << "Integrando queda radial da partícula (r₀ = 15M)..." << std::endl;
    ParticleResult part = integrate_particle(15.0, 0.002, 0.02);
    write_particle_csv(part, outdir);
    // v é reportado como 1 dentro do horizonte (sem observadores estáticos)
    double v_shown = (part.r.back() > r_s) ? part.v.back() : 1.0;
    std::cout << "  Linha de mundo: " << part.tau.size() << " pontos, "
              << "τ_final = " << part.tau.back() << ", "
              << "r_final = " << part.r.back() << ", "
              << "v_final = " << v_shown << std::endl;

    // ============================================================
    // TESTES NUMÉRICOS
    // ============================================================
    std::cout << std::endl;
    std::cout << "=== TESTES NUMÉRICOS ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    bool all_pass = true;

    // ---- Teste 1: parâmetro de impacto crítico ----
    {
        double b_crit_num = find_bcrit();
        double err = std::abs(b_crit_num - b_crit_exact) / b_crit_exact * 100.0;
        bool pass = err < 1.0;
        std::cout << "[Teste 1] Parâmetro de impacto crítico:" << std::endl;
        std::cout << "  b_crit numérico = " << b_crit_num << std::endl;
        std::cout << "  b_crit exato    = " << b_crit_exact << " (3√3 M)" << std::endl;
        std::cout << "  Erro = " << err << " %  (limite 1%)  → "
                  << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) all_pass = false;
    }

    // ---- Teste 2: deflexão em campo fraco ----
    {
        double b = 100.0;
        double alpha_num = measure_deflection(b);
        // 4M/b é só 1ª ordem; inclui 2ª ordem (15π/4)(M/b)² pois em b=100M
        // o termo de 2ª ordem é ~0.00118 (2.95% de 4M/b) — comparar com 4M/b
        // daria erro aparente de ~3% mesmo com integração perfeita.
        double alpha_ana = 4.0 * M / b + (15.0 * M_PI / 4.0) * M * M / (b * b);
        double err = std::abs(alpha_num - alpha_ana) / alpha_ana * 100.0;
        bool pass = err < 2.0;
        std::cout << "[Teste 2] Deflexão em campo fraco (b = 100M):" << std::endl;
        std::cout << "  α numérico = " << alpha_num << " rad" << std::endl;
        std::cout << "  α analítico = " << alpha_ana
                  << " rad  (4M/b + (15π/4)(M/b)²)" << std::endl;
        std::cout << "  Erro = " << err << " %  (limite 2%)  → "
                  << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) all_pass = false;
    }

    // ---- Teste 3: tempo próprio da queda radial ----
    {
        double r0 = 10.0;
        double tau_num = measure_proper_time(r0);
        double tau_ana = (M_PI / 2.0) * std::pow(r0, 1.5) / std::sqrt(2.0 * M);
        double err = std::abs(tau_num - tau_ana) / tau_ana * 100.0;
        bool pass = err < 0.1;
        std::cout << "[Teste 3] Tempo próprio da queda radial (r₀ = 10M):" << std::endl;
        std::cout << "  τ numérico = " << tau_num << std::endl;
        std::cout << "  τ analítico = " << tau_ana << "  ((π/2)·r₀^(3/2)/√(2M))" << std::endl;
        std::cout << "  Erro = " << err << " %  (limite 0.1%)  → "
                  << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) all_pass = false;
    }

    // ---- Teste 4: conservação de energia e momento angular ----
    {
        double max_err = 0.0;
        bool pass = check_photon_conservation(8.0, max_err);
        std::cout << "[Teste 4] Conservação de energia/m. angular (fóton, b=8):" << std::endl;
        std::cout << "  Erro máximo = " << max_err << std::endl;
        std::cout << "  Limite = 1e-6  → " << (pass ? "PASS" : "FAIL") << std::endl;
        if (!pass) all_pass = false;
    }

    std::cout << std::endl;
    std::cout << (all_pass ? ">>> TODOS OS TESTES PASSARAM <<<"
                           : ">>> ALGUNS TESTES FALHARAM <<<") << std::endl;

    return all_pass ? 0 : 1;
}
