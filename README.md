# assimetrias-dalitz-ml

Projeto 2, continuação de `assimetrias-b-mc`. É um estudo metodológico de assimetrias de CP no plano de Dalitz de B± → K±π+π−, com um modelo de brinquedo de três ondas, comparando três métodos:

- ajuste de amplitudes não binado (q_CP);
- método binado de Miranda (S_CP);
- classificador supervisionado (T_ML).

O plano completo está em `../plano_projeto2_v2.pdf`.

## Estado: Marco M0

| Módulo (`src/dalitzcp/`) | Conteúdo | Marco |
|---|---|---|
| `config.py` | lê `config.json` com caminho relativo à raiz do repositório | M0 |
| `sementes.py` | um fluxo `SeedSequence` por etapa e por pseudoexperimento | M0 |
| `cinematica.py` | limites do Dalitz, espaço de fase uniforme | M0, completo no M1 |
| `amplitudes.py` | onda S, R(L=1) em s_ππ, K*(892) em s_Kπ; Zemach, largura constante | M0; Blatt–Weisskopf no M1 |
| `eficiencia.py` | ε±(x) = (1 ± A_det) ε0(x), com ε0 conhecido | M0 |
| `geracao.py` | aceitação-rejeição, com erro se o majorante for violado | M0 |
| `verossimilhanca.py` | Eq. 3, Φ por MC, BFGS com várias partidas, iminuit, q_CP | M0, completo no M2 |
| `estatisticas.py` | S_j ponderado (Eq. 5), conversão para A_fís (Eq. 6) | M0; k-d tree e T_ML no M3 |

Os testes (`tests/`) usam referências independentes do código testado:

- quadrivetores reconstruídos no repouso do B;
- ângulo de helicidade para os fatores de Zemach;
- a forma fechada da Eq. 2;
- a soma de tanh para a Eq. 6;
- o iminuit para o ajuste.

Para confirmar que os testes detectam erros, introduzi de propósito quatro bugs: massa errada no limite, convenção de Δφ invertida, correção de massa do Zemach removida e A_det dependente de x. Cada um faz pelo menos um teste falhar.

## Uso

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
OMP_NUM_THREADS=1 .venv/bin/pytest
OMP_NUM_THREADS=1 .venv/bin/python scripts/piloto_custo.py 1000   # grava resultados/piloto_custo.json
```

Use sempre `OMP_NUM_THREADS=1` ao rodar pseudoexperimentos em paralelo. Com dois processos usando 16 threads de BLAS cada, um ajuste foi de 11 ms para 1,3 s.

## Próximo marco (M1, semanas 2–4)

- Blatt–Weisskopf e Γ(s), mantendo o teste da Eq. 2 abaixo de 1e-12.
- Frações de ajuste e interferências somando 100 ± 0,1 %.
- Figuras F1 (modelo e A_fís no Dalitz) e F2 (transição Δδ0 = 0 → π/2).
- `verificacao.csv` gerada a partir dos testes.
