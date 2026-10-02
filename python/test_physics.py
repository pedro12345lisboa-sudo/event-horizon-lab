#!/usr/bin/env python3
"""
Verificação dos artefatos da Etapa 3.

Confere que o pipeline C++ → CSV → PNG/GIF realmente produziu imagem
(> 50 kB, não branca, com quadros) e que os CSVs têm o conteúdo físico
esperado. Sai com código 1 se algo falhar.

Uso:  python3 blackhole/python/test_physics.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                 # blackhole/
OUT = ROOT / "out"
sys.path.insert(0, str(HERE))
import visualize as viz             # noqa: E402  (traz B_VALUES + o fix 3D)

R_S = viz.R_S
B_CRIT = viz.B_CRIT
B_VALUES = viz.B_VALUES

_min_bytes = 50 * 1024             # 50 kB
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def nonwhite_ratio(im):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    return float((a.sum(axis=2) < 3 * 245).mean())


def main():
    print("=== Verificação dos artefatos (Etapa 3) ===")

    # ------------------------------------------------------------------ 1
    print("[1] Arquivos de saída")
    png = OUT / "painel_etapa2.png"
    gif = OUT / "animacao.gif"
    check("painel_etapa2.png existe", png.is_file(), f"{png}")
    check("animacao.gif existe", gif.is_file(), f"{gif}")
    rays = sorted(OUT.glob("ray_*.csv"))
    check("15 ray_*.csv", len(rays) == 15, f"{len(rays)} encontrados")
    pcsv = OUT / "particle_worldline.csv"
    check("particle_worldline.csv existe", pcsv.is_file())

    # ------------------------------------------------------------------ 2
    print("[2] Imagens maiores que 50 kB e não brancas")
    if png.is_file():
        check("PNG > 50 kB", png.stat().st_size > _min_bytes,
              f"{png.stat().st_size / 1024:.0f} kB")
        im = Image.open(png)
        ratio = nonwhite_ratio(im)
        check("PNG não é uma tela em branco", ratio > 0.02,
              f"{ratio * 100:.2f}% dos pixels não-brancos")
        check("PNG é RGBA/RGB válida", im.size[0] >= 800, f"{im.size}")

    # ------------------------------------------------------------------ 3
    print("[3] GIF de animação")
    if gif.is_file():
        check("GIF > 50 kB", gif.stat().st_size > _min_bytes,
              f"{gif.stat().st_size / 1024 / 1024:.2f} MB")
        g = Image.open(gif)
        n = getattr(g, "n_frames", 1)
        check("GIF tem >= 60 quadros", n >= 60, f"{n} quadros")
        worst = 1.0
        for k in (0, n // 3, 2 * n // 3, n - 1):
            g.seek(k)
            worst = min(worst, nonwhite_ratio(g))
        check("todos os quadros testados são desenhados", worst > 0.01,
              f"pior fração não-branca = {worst * 100:.2f}%")

    # ------------------------------------------------------------------ 4
    print("[4] Geodésicas de fóton (CSV)")
    capturados = set()
    for i, (path, b) in enumerate(zip(rays, B_VALUES)):
        with open(path, newline="") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            rows = list(rd)
        if i == 0:
            check("cabeçalho x,y,z,t", cols == ["x", "y", "z", "t"], str(cols))
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        t = np.array([float(r["t"]) for r in rows])
        r = np.hypot(x, y)
        if i == 0:
            check("t monocómodo não-decrescente", bool(np.all(np.diff(t) >= 0)))
            check("fóton nasce longe (r > 1e5)", r[0] > 1e5, f"r₀ = {r[0]:.3e}")
        if r.min() < R_S + 0.1:
            capturados.add(round(b, 6))

    esperado = {round(abs(b), 6) for b in B_VALUES if abs(b) < B_CRIT}
    obtidos_pos = {c for c in capturados if c > 0}
    check("captura/escape bate com b < 3√3",
          obtidos_pos == {round(e, 6) for e in esperado},
          f"capturados = {sorted(obtidos_pos)} vs esperado {sorted(esperado)}")

    # ------------------------------------------------------------------ 5
    print("[5] Linha de mundo da partícula")
    with open(pcsv, newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    check("cabeçalho completo",
          cols == ["tau", "r", "v", "t", "v_ef", "dtau_dt", "redshift"], str(cols))
    tau = np.array([float(r["tau"]) for r in rows])
    r = np.array([float(r["r"]) for r in rows])
    v = np.array([float(r["v"]) for r in rows])
    t = np.array([float(r["t"]) for r in rows])
    vef = np.array([float(r["v_ef"]) for r in rows])

    check("r cai monotonicamente", bool(np.all(np.diff(r) < 0)),
          f"r: {r[0]:.1f} → {r[-1]:.3f}")
    check("t de Schwarzschild nunca retrocede", bool(np.all(np.diff(t) >= 0)),
          f"t: {t[0]:.1f} → {t[-1]:.1f}")
    check("velocidade local 0 ≤ v ≤ 1", bool(np.all((v >= 0) & (v <= 1 + 1e-12))),
          f"v ∈ [{v.min():.3f}, {v.max():.3f}]")
    check("v_EF cresce (tempo EF é regular)", bool(np.all(np.diff(vef) > 0)))

    tau_ana = (np.pi / 2.0) * 15.0 ** 1.5 / np.sqrt(2.0)
    err = abs(tau[-1] - tau_ana) / tau_ana * 100.0
    check("τ_final (r₀=15) em até 1% do analítico", err < 1.0,
          f"τ = {tau[-1]:.4f} vs {tau_ana:.4f}  ({err:.4f}%)")

    # ------------------------------------------------------------------ 6
    print("[6] Resultado global")
    failed = _results.count(False)
    total = len(_results)
    print(f"\n  {total - failed}/{total} verificações passaram")
    print(">>> TUDO OK <<<" if failed == 0 else ">>> HÁ FALHAS <<<")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
