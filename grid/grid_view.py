#!/usr/bin/env python3
"""
Grade de espaço-tempo curvo em 3D — visualização com ModernGL.

A janela mostra um reticulado de linhas luminosas ("tecido do espaço-tempo")
que é deformado no *vertex shader* a partir das massas da cena. Neste módulo
vivem:

  * `build_lattice`  — geração da grade em numpy (rede 3D ou lençol);
  * `OrbitCamera`    — câmera orbital (arrastar com o mouse, zoom na roda);
  * `Scene`          — shaders, VAOs e desenho (usado pela janela E pelo modo
                       offscreen, para que o PNG seja idêntico ao que se vê);
  * `render_offscreen` — renderiza um quadro em fbo próprio e salva PNG;
  * `GridApp`        — a janela interativa (moderngl_window + pyglet).

Etapa 1 (estado atual): grade 3D ESTÁTICA (u_num_masses == 0), câmera orbital
e captura offscreen verificada por test_grid.py.

Uso:
    python grid_view.py                       # janela interativa
    python grid_view.py --preset leve         # 16 linhas por eixo
    python grid_view.py --mode lencol         # grade 2D no plano
    python grid_view.py --offscreen out/x.png # renderiza e sai
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SHADER_DIR = HERE / "shaders"
OUT_DIR = HERE / "out"

# Presets de densidade: nº de nós por eixo (16 = leve, 24 = padrão, 40 = alto).
# Nesta máquina o padrão é "alto" (RTX 5060, meta de 60 FPS).
PRESETS = {"leve": 16, "padrao": 24, "alto": 40}
MODES = {"rede": 0, "lencol": 1}

EXTENT = 6.0                       # meia-extensão da grade em unidades de cena
GRID_COLOR = (0.75, 1.0, 0.30)     # amarelo-esverdeado
GRID_ALPHA = 0.40                  # brilho base de uma linha isolada
FADE_NEAR_K = 0.9                  # fade começa a essa distância (em half)
FADE_FAR_K = 2.0                   # linhas somem a essa distância (em half)
MASS_K = 1.0                       # constante de acoplamento da deformação
MASS_EPS = 0.35                    # suavização do núcleo (unidades de cena)
MAX_DISP = 1.6                     # clamp do deslocamento (unidades de cena)


# ---------------------------------------------------------------------------
# Geometria da grade (numpy)
# ---------------------------------------------------------------------------
def build_lattice(n: int, half: float = EXTENT, dims: int = 3) -> np.ndarray:
    """Gera os vértices das linhas da grade para desenhar com GL_LINES.

    Cada direção gera n*n polilinhas com n nós cada; cada polilinha vira
    (n - 1) segmentos, e cada segmento vira 2 vértices (o modo LINES do
    ModernGL não usa buffer de índices aqui, então os vértices se repetem).

    dims=3 -> reticulado cúbico (rede 3D): 3 * n² * (n-1) * 2 vértices;
              n = 24 ⇒ 79 488 vértices = 39 744 segmentos.
    dims=2 -> grade só no plano z = 0 (lençol): 2 * n² * (n-1) * 2 vértices.

    Retorna array float32 de forma (N, 3).
    """
    if dims not in (2, 3):
        raise ValueError("dims deve ser 2 (lençol) ou 3 (rede 3D)")
    t = np.linspace(-half, half, n, dtype=np.float32)
    out = []
    for axis in range(dims):
        rest = [a for a in range(3) if a != axis]
        a, b = np.meshgrid(t, t, indexing="ij")
        pts = np.zeros((n * n, n, 3), dtype=np.float32)
        pts[..., axis] = t                    # a linha varia ao longo de `axis`
        # transversais: cada linha (n*n delas) recebe um valor fixo a, b
        pts[..., rest[0]] = a.reshape(-1, 1)  # transversal 1
        if dims == 3:
            pts[..., rest[1]] = b.reshape(-1, 1)  # transversal 2
        # rest[1] permanece 0 no lençol: a grade 2D nasce em z = 0
        seg = np.empty((n * n, n - 1, 2, 3), dtype=np.float32)
        seg[:, :, 0] = pts[:, :-1]
        seg[:, :, 1] = pts[:, 1:]
        out.append(seg.reshape(-1, 3))
    return np.ascontiguousarray(np.concatenate(out, axis=0))


# ---------------------------------------------------------------------------
# Matrizes / câmera orbital
# ---------------------------------------------------------------------------
def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Matriz de visão (convenção GL: vetores-coluna, lookAt clássico)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, np.asarray(up, dtype=np.float64))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Matriz de projeção perspectiva (GL, NDC z ∈ [−1, 1])."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _mvp_bytes(proj: np.ndarray, view: np.ndarray) -> bytes:
    """Combina e serializa no layout coluna-a-coluna que o GL espera."""
    mvp = np.ascontiguousarray((proj @ view).T, dtype="<f4")
    return mvp.tobytes()


class OrbitCamera:
    """Câmera orbital: olha para `target` girando em torno dele.

    yaw/pitch definem a direção, dist a distância. O arrasto do mouse muda
    yaw/pitch e a roda do mouse muda dist (zoom)."""
    MIN_DIST, MAX_DIST = 4.0, 140.0
    MAX_PITCH = 1.45  # ~83°: evita degenerar o vetor `up` em z

    def __init__(self, dist: float = 26.0, yaw: float = -0.9, pitch: float = 0.5,
                 target=(0.0, 0.0, 0.0)):
        self.home = (float(dist), float(yaw), float(pitch))
        self.dist = float(dist)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.target = np.asarray(target, dtype=np.float64)

    def reset(self) -> None:
        self.dist, self.yaw, self.pitch = self.home

    def rotate(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.008
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -self.MAX_PITCH, self.MAX_PITCH))

    def zoom(self, steps: float) -> None:
        self.dist = float(np.clip(self.dist * (0.9 ** steps), self.MIN_DIST, self.MAX_DIST))

    @property
    def eye(self) -> np.ndarray:
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        return self.target + self.dist * np.array([cp * cy, cp * sy, sp])

    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Programa / cena
# ---------------------------------------------------------------------------
def _load_program(ctx: moderngl.Context, name: str) -> moderngl.Program:
    """Compila shaders/grid.<vert|frag>; o shader de teste pode reaproveitar."""
    vs = (SHADER_DIR / f"{name}.vert").read_text()
    fs = (SHADER_DIR / f"{name}.frag").read_text()
    return ctx.program(vertex_shader=vs, fragment_shader=fs)


def _set(prog: moderngl.Program, name: str, value) -> bool:
    """Atribui um uniform se ele existir (compiladores descartam os não usados)."""
    try:
        uni = prog[name]
    except KeyError:
        return False
    if uni is None:
        return False
    uni.value = value
    return True


class Scene:
    """Shaders + buffers da grade e parâmetros de deformação.

    A mesma classe é usada pela janela e pelo renderizador offscreen, para que
    o PNG gerado em `out/` corresponda ao que se vê na tela.
    """

    def __init__(self, ctx: moderngl.Context, preset: str = "alto", mode: str = "rede",
                 half: float = EXTENT):
        self.ctx = ctx
        self.half = float(half)
        self.preset = preset
        self.mode = mode
        self.prog_grid = _load_program(ctx, "grid")
        self.vao = None
        self.vbo = None
        self._build_vao()

    # -- geometria ----------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return PRESETS[self.preset]

    @property
    def dims(self) -> int:
        return 2 if self.mode == "lencol" else 3

    def _build_vao(self) -> None:
        verts = build_lattice(self.n_nodes, self.half, self.dims)
        self.n_vertices = len(verts)
        if self.vbo is not None:
            self.vbo.release()
            self.vao.release()
        self.vbo = self.ctx.buffer(verts)          # float32 (N, 3)
        self.vao = self.ctx.vertex_array(
            self.prog_grid, [(self.vbo, "3f", "in_pos")]
        )

    def set_preset(self, preset: str) -> None:
        if preset != self.preset:
            self.preset = preset
            self._build_vao()

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self._build_vao()

    def toggle_mode(self) -> str:
        self.set_mode("lencol" if self.mode == "rede" else "rede")
        return self.mode

    # -- desenho ------------------------------------------------------------
    def draw(self, cam: OrbitCamera, size) -> None:
        """Desenha um quadro completo (limpa, corpos, grade)."""
        w, h = size
        ctx = self.ctx
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.0, 0.0, 0.0, 1.0)             # fundo preto + depth = far
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"

        # ------------------------------------------------------------------
        # Corpos (esferas) virão aqui na Etapa 2/3, escrevendo profundidade
        # para que as linhas atrás deles sejam ocultadas pelo teste de depth.
        # ------------------------------------------------------------------
        ctx.depth_mask = True

        # -- grade ----------------------------------------------------------
        # Linhas: escrevem NÃO profundidade (depth_mask = False), mas testam.
        # Assim linhas que se cruzam não se escondem umas das outras e a soma
        # aditiva acende o ponto de cruzamento (glow).
        ctx.depth_mask = False
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        proj = perspective(45.0, w / float(h), 0.1, 500.0)
        p = self.prog_grid
        p["u_mvp"].write(_mvp_bytes(proj, cam.view()))
        _set(p, "u_cam_pos", tuple(float(v) for v in cam.eye))

        # Fade por distância da câmera: perto total, longe somendo.
        # FADE_NEAR_K/FADE_FAR_K estão em unidades de meia-extensão da grade.
        d = cam.dist
        _set(p, "u_fade_near", float(max(0.1, d - FADE_NEAR_K * self.half)))
        _set(p, "u_fade_far", float(d + FADE_FAR_K * self.half))
        _set(p, "u_color", GRID_COLOR)
        _set(p, "u_alpha", float(GRID_ALPHA))

        # Deformação (Etapa 1: nenhuma massa ⇒ grade reta)
        _set(p, "u_num_masses", self.n_masses)
        _set(p, "u_k", float(MASS_K))
        _set(p, "u_eps", float(MASS_EPS))
        _set(p, "u_max_disp", float(MAX_DISP))
        _set(p, "u_mode", int(MODES[self.mode]))
        _set(p, "u_funnel", int(self.funnel))
        _set(p, "u_M", float(self.black_hole_mass))
        if self.n_masses:
            _set(p, "u_mass_pos", tuple(float(v) for v in self.mass_pos.reshape(-1)))
            _set(p, "u_mass", tuple(float(v) for v in self.mass))

        self.vao.render(moderngl.LINES)

        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)

    # -- estado (Etapa 1: vazio; ganha massas na Etapa 2) --------------------
    n_masses: int = 0
    mass_pos = np.zeros((8, 3), dtype=np.float32)
    mass = np.zeros(8, dtype=np.float32)
    funnel: int = 0
    black_hole_mass: float = 1.0


# ---------------------------------------------------------------------------
# Render offscreen (PNG)
# ---------------------------------------------------------------------------
def render_offscreen(path: Path, preset: str = "alto", mode: str = "rede",
                     size=(1440, 810), ss: int = 2,
                     dist: float = 26.0, yaw: float = -0.9, pitch: float = 0.5,
                     half: float = EXTENT) -> dict:
    """Renderiza um quadro em um fbo próprio (contexto standalone) e salva PNG.

    `ss` é o supersampling: renderiza em ss*size e reduz com LANCZOS, o que
    suaviza as linhas 1 px sem depender de MSAA.
    Retorna estatísticas do arquivo para os testes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=460)  # OpenGL 4.6
    big = (int(size[0]) * ss, int(size[1]) * ss)
    fbo = ctx.simple_framebuffer(big, components=4)
    fbo.use()

    scene = Scene(ctx, preset=preset, mode=mode, half=half)
    cam = OrbitCamera(dist=dist, yaw=yaw, pitch=pitch)
    scene.draw(cam, big)

    img = Image.frombytes("RGB", big, fbo.read(components=3))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)      # GL lê de baixo para cima
    if ss > 1:
        img = img.resize(size, Image.LANCZOS)
    img.save(path)

    stat = path.stat()
    arr = np.asarray(img)
    return {
        "path": path,
        "bytes": stat.st_size,
        "size": img.size,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "nonblack": float((arr.sum(axis=2) > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Janela interativa
# ---------------------------------------------------------------------------
def run_live(opts: dict) -> None:
    """Abre a janela (moderngl_window + pyglet)."""
    import moderngl_window as mglw

    class GridApp(mglw.WindowConfig):
        title = "Grade de espaco-tempo 3D"
        window_size = (1600, 900)
        gl_version = (4, 6)
        aspect_ratio = None
        resizable = True
        vsync = True
        samples = 4          # MSAA: linhas menos serrilhadas
        cursor = True

        # `opts` (opções da linha de comando) é capturado do escopo de
        # run_live() pelos métodos abaixo — não pode virar atributo da classe
        # porque o corpo da classe tem namespace próprio.

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cam = OrbitCamera(dist=opts["dist"], yaw=opts["yaw"], pitch=opts["pitch"])
            self.scene = Scene(self.ctx, preset=opts["preset"], mode=opts["mode"],
                               half=opts["half"])
            self._t0 = time.time()
            self._frames = 0
            self._fps = 0.0
            self._title_t = 0.0
            self.wnd.title = (
                "Grade de espaço-tempo  |  arrasto=girar  roda=zoom  "
                "M=modo  R=reiniciar  ESC=sair"
            )

        def on_render(self, t: float, frame_time: float) -> None:
            size = (self.wnd.buffer_size[0], self.wnd.buffer_size[1])
            self.scene.draw(self.cam, size)

            # FPS na barra de título (atualiza 2×/s)
            self._frames += 1
            self._title_t += frame_time
            if self._title_t >= 0.5:
                self._fps = self._frames / self._title_t
                self._frames = 0
                self._title_t = 0.0
                self.wnd.title = (
                    f"Grade de espaço-tempo  |  {self._fps:5.1f} FPS  "
                    f"preset={self.scene.preset} modo={self.scene.mode}  "
                    f"vértices={self.scene.n_vertices}  "
                    f"arrasto=girar roda=zoom M=modo R=reiniciar ESC=sair"
                )

        def on_mouse_drag_event(self, x, y, dx, dy) -> None:
            self.cam.rotate(dx, dy)

        def on_mouse_scroll_event(self, x_offset, y_offset) -> None:
            self.cam.zoom(y_offset)

        def on_key_event(self, key, action, modifiers) -> None:
            if action != self.wnd.keys.ACTION_PRESS:
                return
            if key == self.wnd.keys.ESCAPE:
                self.wnd.close()
            elif key == self.wnd.keys.R:
                self.cam.reset()
            elif key == self.wnd.keys.M:
                mode = self.scene.toggle_mode()
                print(f"[modo] {mode}")

    mglw.run_window_config(GridApp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grade de espaço-tempo curvo (ModernGL)")
    p.add_argument("--offscreen", metavar="PNG", type=Path, default=None,
                   help="renderiza um quadro fora de tela e salva no PNG indicado")
    p.add_argument("--preset", choices=sorted(PRESETS), default="alto",
                   help="densidade da grade (padrão: alto = 40 nós/efeixo)")
    p.add_argument("--mode", choices=sorted(MODES), default="rede",
                   help="rede 3D ou lençol 2D")
    p.add_argument("--size", nargs=2, type=int, default=[1440, 810], metavar=("W", "H"),
                   help="resolução do PNG offscreen")
    p.add_argument("--ss", type=int, default=2, help="supersampling offscreen (padrão 2)")
    p.add_argument("--dist", type=float, default=26.0, help="distância da câmera")
    p.add_argument("--yaw", type=float, default=-0.9, help="azimute da câmera (rad)")
    p.add_argument("--pitch", type=float, default=0.5, help="elevação da câmera (rad)")
    p.add_argument("--half", type=float, default=EXTENT, help="meia-extensão da grade")
    return p


def main(argv=None) -> int:
    # Nosso parser aceita os argumentos nossos e deixa os do moderngl_window
    # (--window, --vsync ...) passar adiante via parse_known_args.
    opts, rest = build_parser().parse_known_args(argv)
    values = dict(preset=opts.preset, mode=opts.mode, dist=opts.dist,
                  yaw=opts.yaw, pitch=opts.pitch, half=opts.half)

    if opts.offscreen is not None:
        stat = render_offscreen(opts.offscreen, preset=opts.preset, mode=opts.mode,
                                size=tuple(opts.size), ss=max(1, opts.ss),
                                dist=opts.dist, yaw=opts.yaw, pitch=opts.pitch,
                                half=opts.half)
        print(f"PNG: {stat['path']}  {stat['bytes'] / 1024:.0f} kB  "
              f"{stat['size'][0]}x{stat['size'][1]}  "
              f"pixels [{stat['min']}..{stat['max']}]  "
              f"não-preto = {stat['nonblack'] * 100:.1f}%")
        return 0

    # A janela lê sys.argv com o parser próprio do moderngl_window: escondemos
    # os nossos argumentos para ele não reclamar de opções desconhecidas.
    rest = [a for a in rest if a != "--no-render"]   # flag só do test_grid.py
    sys.argv = [sys.argv[0]] + rest
    run_live(values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
