"""Fluxos de números aleatórios independentes por etapa e por pseudoexperimento.

Cada sorteio usa SeedSequence(semente_mae, spawn_key=(etapa, indice)). Assim o pseudoexperimento i
de uma etapa é reprodutível isoladamente (útil para paralelizar) e nenhum fluxo é reutilizado entre
etapas, que foi um dos problemas do Projeto 1.
"""

import numpy as np

ETAPAS = {
    "normalizacao": 0,
    "piloto": 1,
    "H1": 2,
    "H1_nulo": 3,
    "H2": 4,
    "H3": 5,
    "referencia": 6,
    "testes": 99,
}


def gerador(semente_mae, etapa, indice=0):
    if etapa not in ETAPAS:
        raise KeyError(f"etapa desconhecida: {etapa!r}; registre-a em ETAPAS")
    ss = np.random.SeedSequence(semente_mae, spawn_key=(ETAPAS[etapa], indice))
    return np.random.default_rng(ss)
