#!/usr/bin/env python3
"""
Testes da grade de espaço-tempo (blackhole/grid).

Estilo idêntico a blackhole/python/test_physics.py: verificações com
[PASS]/[FAIL], valores impressos e código de saída 1 se algo falhar.

Uso:  python test_grid.py

Etapa 1 — cobre agora:
  [1] arquivos de shader existem e COMPILAM num contexto OpenGL real;
  [2] contagem de vértices da grade (rede 3D e lençol);
  [3] PNG offscreen gerado, > 50 kB, < 1 MB, pixels não uniformes e não
      todos pretos.

As verificações de deformação (Etapa 2), de órbita/conservação (Etapa 3) e
de funil/vídeo (Etapa 4) entram aqui nas etapas seguintes; a lista final
imprime o que ainda está pendente.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid_view as gv  # noqa: E402

MIN_BYTES = 50 * 1024        # 50 kB
MAX_BYTES = 1024 * 1024      # 1 MB (limite pedido)
PNG_E1 = HERE / "out" / "etapa1_rede.png"

_results = []
_pend = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def pend(name):
    _pend.append(name)
    print(f"  [....] {name}  — pendente")


def main():
    print("=== Testes da grade de espaço-tempo — Etapa 1 ===\n")

    # ------------------------------------------------------------------ 1
    print("[1] Shaders")
    for f in ("grid.vert", "grid.frag"):
        p = HERE / "shaders" / f
        check(f"{f} existe", p.is_file(), str(p))
    try:
        import moderngl
        ctx = moderngl.create_standalone_context(require=460)  # OpenGL 4.6
        prog = gv._load_program(ctx, "grid")
        check("grid.vert + grid.frag compilam", prog is not None,
              f"{ctx.info['GL_RENDERER']} / GL {ctx.info['GL_VERSION']}")
    except Exception as exc:  # noqa: BLE001
        check("grid.vert + grid.frag compilam", False, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ 2
    print("\n[2] Geometria da grade (numpy)")
    n = gv.PRESETS["alto"]
    rede = gv.build_lattice(n, gv.EXTENT, dims=3)
    esperado_rede = 3 * n * n * (n - 1) * 2
    check(f"rede 3D n={n}: {esperado_rede} vértices",
          len(rede) == esperado_rede, f"obtidos {len(rede)}")
    check("rede 3D dentro do cubo [-half, half]",
          rede.min() >= -gv.EXTENT - 1e-6 and rede.max() <= gv.EXTENT + 1e-6,
          f"intervalo [{rede.min():.3f}, {rede.max():.3f}]")
    lencol = gv.build_lattice(n, gv.EXTENT, dims=2)
    esperado_len = 2 * n * n * (n - 1) * 2
    check(f"lençol n={n}: {esperado_len} vértices",
          len(lencol) == esperado_len, f"obtidos {len(lencol)}")
    check("lençol é plano (z = 0 antes da deformação)",
          bool(np.all(lencol[:, 2] == 0.0)), f"z único = {np.unique(lencol[:, 2])}")
    check("dtype float32 (formato do buffer GL)", rede.dtype == np.float32, str(rede.dtype))

    # ------------------------------------------------------------------ 3
    print("\n[3] Render offscreen (PNG)")
    if "--no-render" in sys.argv and PNG_E1.is_file():
        print(f"        reutilizando {PNG_E1} (--no-render)")
    else:
        stat = gv.render_offscreen(PNG_E1, preset="padrao", mode="rede",
                                   size=(1440, 810), ss=2)
        print(f"        gerado: {stat['path']}  {stat['bytes'] / 1024:.0f} kB  "
              f"{stat['size'][0]}x{stat['size'][1]}  "
              f"pixels [{stat['min']}..{stat['max']}]  "
              f"não-preto = {stat['nonblack'] * 100:.1f}%")

    check("PNG existe", PNG_E1.is_file(), str(PNG_E1))
    if PNG_E1.is_file():
        size_b = PNG_E1.stat().st_size
        check("PNG > 50 kB", size_b > MIN_BYTES, f"{size_b / 1024:.1f} kB")
        check("PNG < 1 MB", size_b < MAX_BYTES, f"{size_b / 1024:.1f} kB")

        im = Image.open(PNG_E1).convert("RGB")
        a = np.asarray(im)
        check("resolução mínima 1280x720", im.size[0] >= 1280 and im.size[1] >= 720,
              str(im.size))
        lo, hi = int(a.min()), int(a.max())
        check("extremos de pixel não uniformes", hi > lo, f"min={lo} max={hi}")
        check("não é tudo preto", hi >= 40, f"canal máximo = {hi}")
        frac = float((a.sum(axis=2) > 0).mean())
        check("há conteúdo desenhado (> 1% dos pixels acesos)", frac > 0.01,
              f"{frac * 100:.2f}%")
        # a grade é amarelo-esverdeada: o canal verde deve dominar o vermelho
        g, r = float(a[..., 1].mean()), float(a[..., 0].mean())
        check("linhas amarelo-esverdeadas (G >= R > B)",
              g >= r and r > float(a[..., 2].mean()),
              f"médias R={r:.2f} G={g:.2f} B={float(a[..., 2].mean()):.2f}")

    # ------------------------------------------------------------------ 4
    print("\n[4] Etapas seguintes")
    pend("Etapa 2: deformação numpy == shader (20 pontos, < 1e-4)")
    pend("Etapa 2: PNG com massas fixas deformando a grade")
    pend("Etapa 3: período orbital (< 1%) e conservação E/L (< 1e-6 em 100 órbitas)")
    pend("Etapa 4: funil |z(r)| = 2·sqrt(2M(r-2M)) (< 1e-6)")
    pend("Etapa 4: README + vídeo MP4")

    # ------------------------------------------------------------------ fim
    print("\n[5] Resultado global")
    failed = _results.count(False)
    total = len(_results)
    print(f"  {total - failed}/{total} verificações passaram, {len(_pend)} pendentes")
    print(">>> TUDO OK <<<" if failed == 0 else ">>> HÁ FALHAS <<<")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
