#!/usr/bin/env python3
"""
Etapa 2 — Painéis do buraco negro de Schwarzschild (G = c = M = 1).

Lê os CSVs gerados pelo integrador C++ (blackhole/out/) e produz um PNG
com dois painéis:

  esquerda : as 15 geodésicas de fóton em 3D ao redor da sombra
             (horizonte r = 2M, esfera de fótons r = 3M, ISCO r = 6M)
  direita  : diagrama espaço-tempo da queda radial — compara o tempo de
             Schwarzschild t (diverge no horizonte) com o tempo de
             Eddington-Finkelstein v_EF = t + r + 2M ln|r/2M − 1|
             (cruza o horizonte suavemente, como deve ser).

Uso:  python3 blackhole/python/visualize.py
"""
import csv
import pathlib
import site
from pathlib import Path

# ---------------------------------------------------------------------------
# Correção de ambiente: o Debian instala matplotlib 3.6.3 em /usr/lib e o
# arquivo matplotlib-3.6.3-nspkg.pth pré-registra o pacote de namespace
# "mpl_toolkits" apontando para a versão antiga do sistema — que é incompatível
# com o matplotlib 3.11 do pip (~/.local). Redirecionamos __path__ para o
# mpl_toolkits do pip ANTES de importar o pyplot.
# ---------------------------------------------------------------------------
import mpl_toolkits  # já pré-carregado pelo .pth do sistema

_user_tk = pathlib.Path(site.getusersitepackages()) / "mpl_toolkits"
if _user_tk.is_dir() and str(_user_tk) not in mpl_toolkits.__path__:
    mpl_toolkits.__path__.insert(0, str(_user_tk))

# Garantia extra: se por algum motivo mplot3d ainda resolver para a versão
# antiga do sistema (3.6.3, incompatível), forçamos o caminho do pip na frente
# antes de qualquer import do matplotlib.
if str(_user_tk) in mpl_toolkits.__path__ and mpl_toolkits.__path__[0] != str(_user_tk):
    mpl_toolkits.__path__.remove(str(_user_tk))
    mpl_toolkits.__path__.insert(0, str(_user_tk))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: salva PNG, nunca abre janela
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Constantes físicas (unidades geometricas G = c = M = 1)
# ----------------------------------------------------------------------------
R_S = 2.0        # raio do horizonte de eventos
R_PH = 3.0       # esfera de fótons
R_ISCO = 6.0     # órbita circular estável mais interna
B_CRIT = 3.0 * np.sqrt(3.0)  # parâmetro de impacto crítico ≈ 5.196

OUT = Path("blackhole/out")

# Mesmos valores de b do integrador C++ (geodesics.cpp, função main)
B_VALUES = [2.0, 3.0, 4.0, 5.0, 5.19, 5.21, 6.0, 8.0, 10.0, 15.0, 20.0,
            -2.0, -4.0, -6.0, -10.0]


def load_ray(i):
    """Lê out/ray_XX.csv → (x, y, z) como arrays numpy."""
    xs, ys, zs = [], [], []
    with open(OUT / f"ray_{i:02d}.csv", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def load_particle():
    """Lê out/particle_worldline.csv → dicionário de arrays numpy."""
    cols = {}
    with open(OUT / "particle_worldline.csv", newline="") as f:
        for row in csv.DictReader(f):
            for k, val in row.items():
                cols.setdefault(k, []).append(float(val))
    return {k: np.asarray(v) for k, v in cols.items()}


def draw_sphere(ax, radius, **kw):
    """Esfera de raio `radius` centrada na origem (superfície de plotagem)."""
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    w = np.linspace(0.0, np.pi, 30)
    x = radius * np.outer(np.cos(u), np.sin(w))
    y = radius * np.outer(np.sin(u), np.sin(w))
    z = radius * np.outer(np.ones_like(u), np.cos(w))
    ax.plot_surface(x, y, z, **kw)


def draw_circle(ax, radius, z=0.0, n=200, **kw):
    """Círculo de raio `radius` no plano z = constante."""
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    ax.plot(radius * np.cos(phi), radius * np.sin(phi),
            np.full_like(phi, z), **kw)


def main():
    fig = plt.figure(figsize=(16.0, 7.5), dpi=130, facecolor="white")
    fig.suptitle("Buraco Negro de Schwarzschild — geodésicas e linha de mundo "
                 "(G = c = M = 1)",
                 fontsize=15, fontweight="bold")

    # =====================================================================
    # Painel esquerdo: geodésicas de fóton em 3D
    # =====================================================================
    ax = fig.add_subplot(121, projection="3d")
    ax.set_facecolor("#f7f9fc")

    # -- as 15 geodésicas, coloridas pelo desfecho -------------------------
    for i, b in enumerate(B_VALUES):
        x, y, z = load_ray(i)
        capturado = abs(b) < B_CRIT
        ax.plot(x, y, z,
                color=("#ff4d4d" if capturado else "#29b6f6"),
                lw=1.4 if capturado else 1.6, alpha=0.9, zorder=2)

    # -- o buraco negro ----------------------------------------------------
    # Sombra do horizonte de eventos: esfera opaca preta (nenhuma luz sai).
    draw_sphere(ax, R_S, color="black", alpha=1.0, zorder=5)
    # Esfera de fótons r = 3M: órbitas circulares instáveis de luz.
    for th_deg in (30, 60, 90, 120, 150):
        draw_circle(ax, R_PH * np.sin(np.radians(th_deg)),
                    z=R_PH * np.cos(np.radians(th_deg)),
                    color="darkorange", ls="--", lw=0.9, zorder=4)
    # ISCO r = 6M: borda interna do disco de acreção.
    draw_circle(ax, R_ISCO, color="green", ls=":", lw=1.4, zorder=4)

    # Janela zoomada na região de campo forte (com ±100 a esfera r=2M
    # ficaria com poucos pixels e a dinâmica seria invisível).
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_zlim(-45, 45)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")
    ax.set_zlabel("z / M")
    ax.set_title("(a) Geodésicas de fóton (b = 2 … 20 M)", fontsize=12)
    ax.tick_params(labelsize=7)

    handles = [
        Line2D([], [], color="#ff4d4d", lw=2,
               label=f"fóton capturado  (b < 3√3 = {B_CRIT:.3f}M)"),
        Line2D([], [], color="#29b6f6", lw=2,
               label="fóton escapa  (b > 3√3 M)"),
        Line2D([], [], color="darkorange", ls="--", lw=1.5,
               label="esfera de fótons (r = 3M)"),
        Line2D([], [], color="green", ls=":", lw=1.5, label="ISCO (r = 6M)"),
        Patch(facecolor="black", label="horizonte de eventos (r = 2M)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)

    # =====================================================================
    # Painel direito: diagrama espaço-tempo da queda radial
    # =====================================================================
    ax2 = fig.add_subplot(122)
    p = load_particle()
    tau, r, t, vef = p["tau"], p["r"], p["t"], p["v_ef"]

    # Sombra do interior do horizonte (Schwarzschild t não é definido lá).
    ax2.axhspan(-0.5, R_S, color="0.88", zorder=0)
    ax2.text(4.0, 0.9, "interior do horizonte", fontsize=9, color="0.35")

    # Linhas de interesse em r.
    ax2.axhline(R_S, color="red", ls="--", lw=1.4, zorder=3)
    ax2.axhline(R_PH, color="darkorange", ls="--", lw=1.0, zorder=3)
    ax2.axhline(R_ISCO, color="green", ls=":", lw=1.2, zorder=3)
    ax2.text(118.5, R_S + 0.35, "horizonte r = 2M", color="red",
             fontsize=9, ha="right")
    ax2.text(118.5, R_PH + 0.35, "esfera de fótons", color="darkorange",
             fontsize=9, ha="right")
    ax2.text(118.5, R_ISCO + 0.35, "ISCO", color="green",
             fontsize=9, ha="right")

    # Tempo de Schwarzschild: plotado só onde r ≥ 2M — diverge ao cruzar.
    out = r >= R_S
    ax2.plot(t[out], r[out], color="#1565c0", lw=2.2,
             label="t de Schwarzschild  (diverge no horizonte)", zorder=4)

    # Tempo de Eddington-Finkelstein: cruza o horizonte suavemente.
    ax2.plot(vef, r, color="#2e7d32", lw=2.2,
             label="v de Eddington-Finkelstein  (cruza suavemente)", zorder=5)

    # Marca o instante do cruzamento (v_EF ≈ constante lá).
    ic = np.argmax(r <= R_S)
    ax2.plot([vef[ic]], [R_S], marker="o", ms=7, color="black", zorder=6)
    ax2.annotate("cruzamento\nem v_EF finito",
                 xy=(vef[ic], R_S), xytext=(vef[ic] + 16, 4.4),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.2))

    ax2.set_xlim(0, 120)
    ax2.set_ylim(-0.5, 16)
    ax2.set_xlabel("tempo coordenativo  (unidades de M)")
    ax2.set_ylabel("r / M")
    ax2.set_title("(b) Queda radial da partícula (r₀ = 15M, do repouso)",
                  fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center left", fontsize=9, framealpha=0.95)

    # Rodapé com o dado numérico do cruzamento.
    fig.text(0.5, 0.01,
             f"t cruza o horizonte em ≈ {t[ic]:.1f} M (divergência logarítmica "
             f"não resolvida por passo finito)   |   v_EF no cruzamento = "
             f"{vef[ic]:.2f} M   |   τ_final = {tau[-1]:.2f} M",
             ha="center", fontsize=9, color="0.35")

    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    out_png = OUT / "painel_etapa2.png"
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"PNG salvo: {out_png}  ({out_png.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
