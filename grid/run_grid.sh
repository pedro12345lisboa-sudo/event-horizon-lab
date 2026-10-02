#!/usr/bin/env bash
# Ativa o .venv do projeto, roda os testes da grade e abre a janela.
#
#   ./run_grid.sh                     janela (preset alto, rede 3D)
#   ./run_grid.sh --preset leve       grade mais esparsa
#   ./run_grid.sh --mode lencol       grade 2D (lençol)
#   ./run_grid.sh --offscreen out/x.png   só renderiza o PNG e sai
#   python test_grid.py --no-render   só os testes, reaproveitando o PNG
#
# Os testes rodam SEM argumentos; os argumentos vão só para a janela/PNG.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJETO="$(cd "$HERE/../.." && pwd)"
VENV="$PROJETO/.venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERRO: .venv não encontrado em $VENV" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "venv: $VENV"

# O moderngl cria o contexto de JANELA com dlopen("libGL.so"); este sistema
# só instala libGL.so.1 (o symlink sem versão vem do pacote dev, ex. libgl-dev).
# Em vez de instalar pacote, criamos o symlink aqui dentro do projeto e o
# apontamos pela LD_LIBRARY_PATH. O render offscreen (testes) não precisa.
LIBDIR="$HERE/lib"
mkdir -p "$LIBDIR"
if [ ! -e "$LIBDIR/libGL.so" ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 "$LIBDIR/libGL.so"
    echo "symlink: $LIBDIR/libGL.so -> /usr/lib/x86_64-linux-gnu/libGL.so.1"
fi
export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python "$HERE/test_grid.py"
echo
exec python "$HERE/grid_view.py" "$@"
