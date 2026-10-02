#!/usr/bin/env python3
"""
Etapa 3 — Animação sincronizada das geodésicas (relógio de Schwarzschild).

Relógio único = tempo coordenado t de Schwarzschild:

  * partícula  : t = 0 no instante da liberação (r₀ = 15M, do repouso)
  * cada fóton : t = 0 no instante em que entra na caixa visível
                 (|x| ≤ 45M e |y| ≤ 45M)

Depois desses instantes ambos avançam com o MESMO relógio (dt_anim = dt),
de modo que o quadro mostra o que cada objeto faria num dado instante de
tempo coordenado — não uma "rodada" independente por objeto.

A coluna t de cada fóton vem do C++, integrada a partir de

    dt/dψ = 1 / (b u² (1 - 2Mu)),

que congeladentro do horizonte (1 - 2Mu ≤ 0) — exatamente como o t da
partícula. Consequência física: fótons capturados e a partícula CONGELAM
na esfera r = 2M no tempo de Schwarzschild, que é o que um observador
distante veria (a cruzagem real acontece em t = ∞).

Uso:  python3 blackhole/python/view.py
"""
import csv

import numpy as np

import visualize as viz  # aplica o fix de mpl_toolkits + backend Agg
import matplotlib.pyplot as plt
from PIL import Image

BOX = 45.0          # meia-lado da caixa de visualização
N_FRAMES = 120
DUR_MS = 80         # ms por quadro do GIF

OUT = viz.OUT
B_VALUES = viz.B_VALUES
R_S, R_PH, R_ISCO = viz.R_S, viz.R_PH, viz.R_ISCO
B_CRIT = viz.B_CRIT

COL_CAP = "#ff4d4d"
COL_ESC = "#29b6f6"
COL_PART = "#ffd54f"


def load_ray(i):
    """Lê out/ray_XX.csv → (x, y, z, t)."""
    xs, ys, zs, ts = [], [], [], []
    with open(OUT / f"ray_{i:02d}.csv", newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            ts.append(float(row["t"]))
    return (np.asarray(xs), np.asarray(ys),
            np.asarray(zs), np.asarray(ts))


def _cut_frozen(t, *rest):
    """Corta no primeiro instante em que t deixa de crescer (freeze dentro
    do horizonte). Devolve os recortes alinhados + índice do corte."""
    d = np.diff(t)
    bad = np.flatnonzero(d <= 0.0)
    if not bad.size:
        return (t,) + rest, len(t)
    cut = int(bad[0]) + 1
    return (t[:cut],) + tuple(a[:cut] for a in rest), cut


def prepare():
    """Carrega os CSVs e monta o relógio de animação comum."""
    rays = []
    T_max = 0.0
    for i, b in enumerate(B_VALUES):
        x, y, z, t = load_ray(i)
        # âncora: t = 0 na primeira amostra que entra na caixa visível
        inside = (np.abs(x) <= BOX) & (np.abs(y) <= BOX)
        i0 = int(np.argmax(inside)) if inside.any() else 0
        ta = t - t[i0]
        (ta, x, y, z), cut = _cut_frozen(ta, x, y, z)
        if i0 >= cut:                      # nunca deve acontecer
            i0 = 0
        r = np.hypot(x, y)
        vis = (np.abs(x) <= BOX) & (np.abs(y) <= BOX)
        if vis.any():
            T_max = max(T_max, float(ta[vis].max()))
        rays.append(dict(i0=i0, x=x, y=y, z=z, ta=ta, r=r,
                         color=(COL_ESC if abs(b) > B_CRIT else COL_CAP),
                         label=f"b = {b:g}"))

    p = viz.load_particle()
    (tp, rp, taup), _ = _cut_frozen(p["t"], p["r"], p["tau"])
    T_max = max(T_max, float(tp.max()))
    return rays, tp, rp, taup, T_max


def main():
    rays, tp, rp, taup, T_max = prepare()
    print(f"Relógio de animação: t ∈ [0, {T_max:.1f}] M   "
          f"({N_FRAMES} quadros, {DUR_MS} ms/quadro)")

    fig = plt.figure(figsize=(13.5, 5.8), dpi=95, facecolor="white")
    fig.suptitle("Buraco negro de Schwarzschild — fótons e queda radial "
                 "sincronizados pelo tempo t", fontsize=14, fontweight="bold")

    # =====================================================================
    # Painel (a): 3D — geodésicas ao redor da sombra
    # =====================================================================
    ax = fig.add_subplot(121, projection="3d")
    ax.set_facecolor("#f7f9fc")
    # estáticos
    viz.draw_sphere(ax, R_S, color="black", alpha=1.0, zorder=5)
    for th_deg in (30, 60, 90, 120, 150):
        viz.draw_circle(ax, R_PH * np.sin(np.radians(th_deg)),
                        z=R_PH * np.cos(np.radians(th_deg)),
                        color="darkorange", ls="--", lw=0.9, zorder=4)
    viz.draw_circle(ax, R_ISCO, color="green", ls=":", lw=1.4, zorder=4)

    trails, heads = [], []
    for ray in rays:
        tr, = ax.plot([], [], [], color=ray["color"], lw=1.6,
                      alpha=0.9, zorder=3)
        hd, = ax.plot([], [], [], color=ray["color"], ls="None", marker="o",
                      ms=5.5, mfc=ray["color"], mec="k", mew=0.6, zorder=7)
        trails.append(tr)
        heads.append(hd)
    ptrail, = ax.plot([], [], [], color=COL_PART, lw=2.6, zorder=8)
    phead, = ax.plot([], [], [], color=COL_PART, ls="None", marker="o",
                     ms=9, mfc=COL_PART, mec="k", mew=1.1, zorder=9)

    ax.set_xlim(-BOX, BOX)
    ax.set_ylim(-BOX, BOX)
    ax.set_zlim(-BOX, BOX)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x / M")
    ax.set_ylabel("y / M")
    ax.set_zlabel("z / M")
    ax.set_title("(a) Plano das geodésicas  (partícula cai pelo eixo +z)",
                 fontsize=11)
    ax.tick_params(labelsize=7)
    ax.legend(handles=[
        plt.Line2D([], [], color=COL_CAP, lw=2,
                   label=f"capturado (b < 3√3 = {B_CRIT:.3f}M)"),
        plt.Line2D([], [], color=COL_ESC, lw=2, label="escapa"),
        plt.Line2D([], [], color=COL_PART, lw=2.6, marker="o", ms=7,
                   label="partícula (r₀ = 15M)"),
        plt.Line2D([], [], color="darkorange", ls="--", lw=1.5,
                   label="esfera de fótons (r = 3M)"),
        plt.Line2D([], [], color="green", ls=":", lw=1.5, label="ISCO (r = 6M)"),
    ], loc="upper right", fontsize=7.5, framealpha=0.92)

    # =====================================================================
    # Painel (b): r(t) de tudo, com cursor no tempo corrente
    # =====================================================================
    ax2 = fig.add_subplot(122)
    ax2.axhspan(-0.5, R_S, color="0.88", zorder=0)
    ax2.axhline(R_S, color="red", ls="--", lw=1.3, zorder=3)
    ax2.axhline(R_PH, color="darkorange", ls="--", lw=1.0, zorder=3)
    ax2.axhline(R_ISCO, color="green", ls=":", lw=1.2, zorder=3)
    ax2.text(0.6, R_S + 1.2, "interior do horizonte (t congela)", color="red",
             fontsize=8.5)

    for ray in rays:
        ax2.plot(ray["ta"], ray["r"], color=ray["color"], lw=1.0,
                 alpha=0.55, zorder=2)
    ax2.plot(tp, rp, color="#1565c0", lw=2.4, zorder=5,
             label="partícula: r(t)")

    cursor = ax2.axvline(0.0, color="#c62828", lw=1.8, zorder=6)
    pdot, = ax2.plot([], [], ls="None", marker="o", ms=9, mfc=COL_PART,
                     mec="k", mew=1.1, zorder=7)
    _nan = np.full(len(rays), np.nan)
    bdot = ax2.scatter(_nan, _nan, s=30, c=[r["color"] for r in rays],
                       edgecolors="k", linewidths=0.5, zorder=6)
    tbox = ax2.text(0.985, 0.965, "", transform=ax2.transAxes, ha="right",
                    va="top", fontsize=11, fontweight="bold",
                    bbox=dict(fc="white", ec="0.55", alpha=0.92))

    ax2.set_xlim(0.0, T_max * 1.02)
    ax2.set_ylim(0.0, 55.0)
    ax2.set_xlabel("tempo de animação  t  (unidades de M)")
    ax2.set_ylabel("r / M")
    ax2.set_title("(b) Raio vs tempo — mesmo relógio para tudo", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="center left", fontsize=9, framealpha=0.95)

    fig.text(0.5, 0.012,
             "t = 0 no evento de cada objeto (partícula liberada em r = 15M; "
             "fóton ao entrar na caixa).  dt_anim = dt de Schwarzschild.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))

    # =====================================================================
    # Loop de animação
    # =====================================================================
    frames = []
    times = np.linspace(0.0, T_max, N_FRAMES)
    for n, t_now in enumerate(times):
        # -- fótons ------------------------------------------------------
        pts = np.empty((len(rays), 2))
        for i, ray in enumerate(rays):
            k = int(np.searchsorted(ray["ta"], t_now, side="right"))
            k = min(max(k, ray["i0"] + 1), len(ray["ta"]))
            sl = slice(ray["i0"], k)
            trails[i].set_data_3d(ray["x"][sl], ray["y"][sl], ray["z"][sl])
            heads[i].set_data_3d(ray["x"][k - 1:k], ray["y"][k - 1:k],
                                 ray["z"][k - 1:k])
            pts[i] = (ray["ta"][k - 1], ray["r"][k - 1])

        # -- partícula (cai pelo eixo +z) -------------------------------
        kp = int(np.searchsorted(tp, t_now, side="right"))
        kp = min(max(kp, 1), len(tp))
        zp = rp[:kp]
        ptrail.set_data_3d(np.zeros(kp), np.zeros(kp), zp)
        phead.set_data_3d(np.array([0.0]), np.array([0.0]),
                          np.array([zp[-1]]))

        # -- painel (b) --------------------------------------------------
        cursor.set_data([t_now, t_now], [0.0, 1.0])
        pdot.set_data([t_now], [rp[kp - 1]])
        bdot.set_offsets(pts)
        tbox.set_text(f"t = {t_now:6.1f} M")

        # -- captura -----------------------------------------------------
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB")
                      .convert("P", palette=Image.ADAPTIVE, colors=256))
        if (n + 1) % 20 == 0:
            print(f"  quadro {n + 1}/{N_FRAMES}")

    plt.close(fig)

    out_gif = OUT / "animacao.gif"
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=DUR_MS, loop=0, optimize=True)
    print(f"GIF salvo: {out_gif}  "
          f"({out_gif.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{len(frames)} quadros)")


if __name__ == "__main__":
    main()
