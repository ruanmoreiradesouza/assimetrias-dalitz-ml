"""Geração de eventos B− e B+ por aceitação-rejeição sobre ε(x)|M(x)|²."""

import numpy as np

from .amplitudes import intensidade, matriz_amplitudes
from .cinematica import espaco_fase
from .eficiencia import eficiencia

FOLGA_MAJORANTE = 1.5


def densidade(x, theta, carga, m, cfg):
    return eficiencia(x, carga, m, cfg) * intensidade(matriz_amplitudes(x, m, cfg), theta, carga)


def gerar_carga(rng, theta, n, carga, m, cfg):
    """n eventos de uma carga. O majorante vem do primeiro lote com folga de 50 %; se algum ponto o
    ultrapassar, a amostra estaria distorcida, então o erro é explícito em vez de silencioso."""
    majorante = None
    partes, total = [], 0
    while total < n:
        x = espaco_fase(rng, max(20 * n, 10_000), m)
        d = densidade(x, theta, carga, m, cfg)
        if majorante is None:
            majorante = FOLGA_MAJORANTE * d.max()
        if d.max() > majorante:
            raise RuntimeError("densidade acima do majorante: aumente FOLGA_MAJORANTE")
        aceitos = x[rng.uniform(0, majorante, len(x)) < d]
        partes.append(aceitos)
        total += len(aceitos)
    return np.concatenate(partes)[:n]


def gerar_amostra(rng, theta, n_total, m, cfg):
    """Amostra não-estendida: n_total/2 eventos por carga. Devolve {carga: posições N×2}."""
    return {carga: gerar_carga(rng, theta, n_total // 2, carga, m, cfg) for carga in (-1, +1)}
