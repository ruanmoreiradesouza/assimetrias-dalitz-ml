"""Piloto do orçamento computacional (Seção 6 do plano v2), agora sobre o pacote dalitzcp.

Uso:  OMP_NUM_THREADS=1 python scripts/piloto_custo.py [n_toys]
Grava resultados/piloto_custo.json com metadados (versão, semente, data).
"""

import datetime
import json
import math
import sys
import time

import numpy as np
from scipy import stats

import dalitzcp
from dalitzcp.cinematica import Massas
from dalitzcp.config import RAIZ, carregar_config, theta_nominal
from dalitzcp.geracao import gerar_amostra
from dalitzcp.sementes import gerador
from dalitzcp.verossimilhanca import ajustar, ajustar_minuit, amplitudes_da_amostra, matriz_phi, q_cp


def main(n_toys):
    cfg = carregar_config()
    m = Massas.de_config(cfg)
    semente = cfg["semente_mae"]
    theta = theta_nominal(cfg)
    N = cfg["N_eventos"]
    phis = matriz_phi(gerador(semente, "normalizacao"), m, cfg)

    amps = amplitudes_da_amostra(gerar_amostra(gerador(semente, "piloto", 0), theta, N, m, cfg), m, cfg)
    t = time.perf_counter()
    r = ajustar(amps, phis, theta, gerador(semente, "piloto", 1), partidas=1)
    t_scipy = time.perf_counter() - t
    t = time.perf_counter()
    mi = ajustar_minuit(amps, phis, theta)
    t_minuit = time.perf_counter() - t
    conc = float(np.max(np.abs(r.x - np.array(mi.values)) / np.array(mi.errors)))
    print(f"N={N}: ajuste scipy {1e3*t_scipy:.0f} ms | iminuit {1e3*t_minuit:.0f} ms | "
          f"max|Δθ|/σ = {conc:.1e} | Δφ̂ = {mi.values['dphi']:.3f} ± {mi.errors['dphi']:.3f}")

    theta0 = list(theta)
    theta0[4] = 0.0
    qs = []
    t = time.perf_counter()
    for i in range(n_toys):
        rng = gerador(semente, "H3", i)
        a = amplitudes_da_amostra(gerar_amostra(rng, theta0, N, m, cfg), m, cfg)
        qs.append(q_cp(a, phis, theta0, rng)[0])
    t_toy = (time.perf_counter() - t) / max(n_toys, 1)
    qs = np.array(qs)
    taxa = float(np.mean(qs > stats.chi2.ppf(0.95, 1))) if n_toys else math.nan
    print(f"H0, {n_toys} toys: {t_toy:.2f} s/toy | rejeição com χ²₁(95%) = {100*taxa:.1f}% "
          f"(esperado 5,0 ± {100*math.sqrt(0.0475/max(n_toys, 1)):.1f}%) | média q_CP = {qs.mean():.2f}")

    saida = {
        "versao": dalitzcp.__version__,
        "data": datetime.datetime.now().isoformat(timespec="seconds"),
        "semente_mae": semente,
        "N": N,
        "ms_ajuste_scipy": 1e3 * t_scipy,
        "ms_ajuste_minuit": 1e3 * t_minuit,
        "concordancia_max_sigma": conc,
        "n_toys_H0": n_toys,
        "s_por_toy": t_toy,
        "taxa_rejeicao_chi2_1": taxa,
        "media_qcp": float(qs.mean()) if n_toys else math.nan,
    }
    (RAIZ / "resultados").mkdir(exist_ok=True)
    with open(RAIZ / "resultados" / "piloto_custo.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
