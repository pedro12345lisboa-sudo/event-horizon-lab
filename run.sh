#!/usr/bin/env bash
# ============================================================
# run.sh — pipeline completo do buraco negro de Schwarzschild
# ============================================================
# 1. compila o integrador de geodésicas (C++17, RK4)
# 2. roda os 4 testes numéricos e gera os CSVs
# 3. gera o painel estático (PNG)  — Etapa 2
# 4. gera a animação sincronizada (GIF) — Etapa 3
# 5. verifica os artefatos gerados
#
# Sem dependências externas: g++ + python3 (numpy, matplotlib, Pillow).
# ============================================================
set -euo pipefail

# Raiz do projeto (este script fica em blackhole/)
cd "$(dirname "$0")/.."

echo "=========================================================="
echo " Buraco negro de Schwarzschild — pipeline completo"
echo "=========================================================="
mkdir -p blackhole/out

echo
echo "--- [1/5] Compilando geodesics.cpp (C++17, RK4) ---"
g++ -std=c++17 -O2 -Wall -Wextra blackhole/src/geodesics.cpp \
    -o blackhole/geodesics -lm
echo "ok: blackhole/geodesics"

echo
echo "--- [2/5] Integrando geodésicas + testes numéricos ---"
./blackhole/geodesics

echo
echo "--- [3/5] Painel estático (painel_etapa2.png) ---"
python3 blackhole/python/visualize.py

echo
echo "--- [4/5] Animação sincronizada (animacao.gif) ---"
python3 blackhole/python/view.py

echo
echo "--- [5/5] Verificando artefatos ---"
python3 blackhole/python/test_physics.py

echo
echo "=========================================================="
echo " Saídas em blackhole/out/:"
ls -lh blackhole/out/ | sed 's/^/   /'
echo "=========================================================="
