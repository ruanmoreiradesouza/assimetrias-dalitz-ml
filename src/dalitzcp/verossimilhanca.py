"""Verossimilhança não-binada e não-estendida (Eq. 3 do plano) e estatística q_CP.

ln L(θ) = Σ_λ [ Σ_i ln|M_λ(x_i; θ)|² − N_λ ln(c_λ† Φ_λ c_λ) ],  Φ_λ,rs = ∫ ε_λ A_r A_s* dx.
O termo ln ε(x_i) não depende de θ e é omitido. A eficiência entra só por Φ.
"""

import math

import numpy as np
from iminuit import Minuit
from scipy import optimize

from .amplitudes import coeficientes, matriz_amplitudes
from .cinematica import espaco_fase
from .eficiencia import eficiencia

NOMES = ("c2", "delta2", "c3", "delta3", "dphi")


def matriz_phi(rng, m, cfg, n_mc=None):
    """Φ_λ estimada por Monte Carlo uniforme no Dalitz (normalização comum, irrelevante para o ajuste)."""
    n_mc = n_mc or cfg["n_mc_normalizacao"]
    x = espaco_fase(rng, n_mc, m)
    a = matriz_amplitudes(x, m, cfg)
    phis = {}
    for carga in (-1, +1):
        w = eficiencia(x, carga, m, cfg)
        phis[carga] = (a * w[:, None]).T @ a.conj() / n_mc
    return phis


def amplitudes_da_amostra(amostra, m, cfg):
    return {carga: matriz_amplitudes(x, m, cfg) for carga, x in amostra.items()}


def nll(theta, amps, phis):
    total = 0.0
    for carga in (-1, +1):
        c = coeficientes(theta, carga)
        m2 = np.abs(amps[carga] @ c) ** 2
        norma = np.real(c @ phis[carga] @ c.conj())
        total -= np.sum(np.log(m2)) - len(m2) * np.log(norma)
    return total


def ajustar(amps, phis, theta0, rng, fixar_dphi=False, partidas=3):
    """BFGS do scipy com várias partidas (fases sorteadas a partir da 2ª). Devolve o melhor resultado,
    com r.x sempre de tamanho 5 (Δφ = 0 quando fixado)."""
    melhor = None
    for k in range(partidas):
        t0 = np.array(theta0, dtype=float)
        if k:
            t0[[1, 3]] = rng.uniform(-math.pi, math.pi, 2)
            t0[4] = 0.0 if fixar_dphi else rng.uniform(-1, 1)
        if fixar_dphi:
            r = optimize.minimize(lambda t: nll(np.append(t, 0.0), amps, phis), t0[:4], method="BFGS")
            r.x = np.append(r.x, 0.0)
        else:
            r = optimize.minimize(nll, t0, args=(amps, phis), method="BFGS")
        if melhor is None or r.fun < melhor.fun:
            melhor = r
    return melhor


def ajustar_minuit(amps, phis, theta0):
    """Referência externa: MIGRAD + HESSE do iminuit, partindo do mesmo θ0."""
    m = Minuit(lambda c2, delta2, c3, delta3, dphi: nll([c2, delta2, c3, delta3, dphi], amps, phis),
               *theta0, name=NOMES)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return m


def q_cp(amps, phis, theta0, rng, partidas=3):
    """q_CP = 2[ln L(θ̂) − ln L(θ̂_{Δφ=0})] ≥ 0."""
    livre = ajustar(amps, phis, theta0, rng, partidas=partidas)
    nulo = ajustar(amps, phis, theta0, rng, fixar_dphi=True, partidas=partidas)
    return max(2.0 * (nulo.fun - livre.fun), 0.0), livre, nulo
