# Buraco negro de Schwarzschild — geodésicas em C++/Python

Simulação numérica do espaço-tempo de Schwarzschild com **G = c = M = 1**
(unidades geométricas, logo `r_s = 2M = 2`): geodésicas de fóton (luz) e a
queda radial de uma partícula a partir do repouso, integradas com **RK4 em
C++17** e visualizadas em **Python**.

## Como rodar

```bash
./blackhole/run.sh
```

Compila, integra, gera as imagens e verifica os artefatos. Saída em
`blackhole/out/`.

Dependências: `g++` (C++17) e `python3` com `numpy`, `matplotlib`, `Pillow`.
Nada além disso — sem pandas, sem frameworks.

## Física

| Grandeza | Valor |
|---|---|
| Raio de Schwarzschild | `r_s = 2M = 2` |
| Esfera de fótons | `r = 3M = 3` |
| ISCO | `r = 6M = 6` |
| Parâmetro de impacto crítico | `b_crit = 3√3 M ≈ 5.19615` |

**Fóton** (órbita em `u = 1/r`, equatorial):

```
d²u/dφ² + u = 3Mu²
(du/dψ)² = 1/b² − u²(1 − 2Mu)        ψ = π − φ
dt/dψ    = 1 / (b u² (1 − 2Mu))      tempo coordenado
```

É capturado se `b < b_crit`, escapa caso contrário. `dt/dψ` **congela**
dentro do horizonte, onde `t` deixa de ser temporal.

**Partícula** (queda radial do repouso em `r₀ = 15M`), estado
`(r, v_r, v_local, t, v_EF)`:

```
d²r/dτ² = −M/r²
dv/dτ   = M/(ε r²)                                ε = √(1 − 2M/r₀)
dt/dτ   = ε/(1 − 2M/r)     (só fora do horizonte)
dv_EF/dτ = 1/(ε + √(ε² − (1 − 2M/r)))
```

* `v_local` é a velocidade medida por observadores estáticos (0 no repouso,
  → 1 no horizonte); a forma acima é regular, ao contrário da identidade
  `v = |dr/dτ|/ε`, que é singular no repouso.
* `v_EF = t + r + 2M ln|r/2M − 1|` é o tempo de **Eddington–Finkelstein**,
  que cruza o horizonte sem divergir (75.470 no cruzamento), ao contrário
  de `t`, que diverge.

## Estrutura

```
blackhole/
├── src/geodesics.cpp      integrador RK4 + 4 testes numéricos (C++17)
├── python/visualize.py    Etapa 2 — painel estático (PNG)
├── python/view.py         Etapa 3 — animação sincronizada (GIF)
├── python/test_physics.py verificação dos artefatos gerados
├── run.sh                 pipeline completo
├── LICENSE                MIT
└── out/                   CSVs + PNG + GIF (gerados)
```

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `out/ray_XX.csv` | 15 geodésicas de fóton, colunas `x,y,z,t` |
| `out/particle_worldline.csv` | `tau,r,v,t,v_ef,dtau_dt,redshift` |
| `out/painel_etapa2.png` | geodésicas 3D + diagrama espaço-tempo |
| `out/animacao.gif` | animação com relógio único (tempo `t` de Schwarzschild) |

## Animação sincronizada

O relógio é o **tempo coordenado `t`**:

* a partícula tem `t = 0` no instante da liberação (`r₀ = 15M`, do repouso);
* cada fóton tem `t = 0` no instante em que entra na caixa visível
  (`|x| ≤ 45M` e `|y| ≤ 45M`).

Depois disso ambos avançam com o mesmo relógio (`dt_anim = dt`), então um
quadro mostra o que cada objeto faria num dado instante de tempo
coordenado. Como `t` congela dentro do horizonte, fótons capturados e a
partícula **congelam na esfera `r = 2M`** — exatamente o que um observador
distante veria (a cruzagem real ocorre em `t = ∞`).

## Testes numéricos

O binário roda 4 testes e só retorna 0 se todos passarem:

| Teste | Comparação com o analítico | Limite |
|---|---|---|
| 1. `b_crit` por bissecção | `3√3 M` | < 1 % |
| 2. Deflexão em campo fraco (`b = 100M`) | `4M/b + (15π/4)(M/b)²` | < 2 % |
| 3. Tempo próprio da queda (`r₀ = 10M`) | `(π/2)·r₀^{3/2}/√(2M)` | < 0.1 % |
| 4. Conservação de energia/m. angular | `1/b²` | < 1e-6 |

`test_physics.py` confere depois que os artefatos realmente existem, passam
de 50 kB, não são telas em branco e têm o conteúdo físico esperado.

## Licença

MIT — ver [`LICENSE`](LICENSE).
