"""Cinemática de B± → K± π+ π− no plano de Dalitz.

Variáveis: x = (s_Kπ, s_ππ), com s_Kπ = m²(K± π∓) e s_ππ = m²(π+ π−). A terceira invariante é
s_Kπ' = m²(K± π±) = M_B² + m_K² + 2 m_π² − s_Kπ − s_ππ.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Massas:
    B: float
    K: float
    pi: float

    @classmethod
    def de_config(cls, cfg):
        m = cfg["massas_GeV"]
        return cls(B=m["B"], K=m["K"], pi=m["pi"])

    @property
    def soma_quadrados(self):
        return self.B**2 + self.K**2 + 2 * self.pi**2


def kallen(x, y, z):
    return x * x + y * y + z * z - 2 * (x * y + x * z + y * z)


def s_kpi_linha(s_kpi, s_pipi, m):
    """m²(K± π±), a invariante dependente."""
    return m.soma_quadrados - s_kpi - s_pipi


def faixa_s_pipi(m):
    return (2 * m.pi) ** 2, (m.B - m.K) ** 2


def faixa_s_kpi(m):
    return (m.K + m.pi) ** 2, (m.B - m.pi) ** 2


def limites_s_kpi(s_pipi, m):
    """Limites de s_Kπ para s_ππ fixo, pelas energias no repouso do par ππ."""
    raiz = np.sqrt(s_pipi)
    e_pi = raiz / 2
    e_k = (m.B**2 - s_pipi - m.K**2) / (2 * raiz)
    p_pi = np.sqrt(np.maximum(e_pi**2 - m.pi**2, 0.0))
    p_k = np.sqrt(np.maximum(e_k**2 - m.K**2, 0.0))
    return (e_pi + e_k) ** 2 - (p_pi + p_k) ** 2, (e_pi + e_k) ** 2 - (p_pi - p_k) ** 2


def dentro(s_kpi, s_pipi, m):
    lo_pp, hi_pp = faixa_s_pipi(m)
    ok = (s_pipi > lo_pp) & (s_pipi < hi_pp)
    lo, hi = limites_s_kpi(np.where(ok, s_pipi, lo_pp + 1e-9), m)
    return ok & (s_kpi >= lo) & (s_kpi <= hi)


def espaco_fase(rng, n, m):
    """n pontos uniformes no plano de Dalitz (o espaço de fase de 3 corpos é plano em (s_Kπ, s_ππ))."""
    lo_kp, hi_kp = faixa_s_kpi(m)
    lo_pp, hi_pp = faixa_s_pipi(m)
    partes, total = [], 0
    while total < n:
        s_kp = rng.uniform(lo_kp, hi_kp, 3 * n)
        s_pp = rng.uniform(lo_pp, hi_pp, 3 * n)
        ok = dentro(s_kp, s_pp, m)
        partes.append(np.column_stack([s_kp[ok], s_pp[ok]]))
        total += int(ok.sum())
    return np.concatenate(partes)[:n]
