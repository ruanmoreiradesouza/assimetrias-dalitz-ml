"""Modelo isobárico de brinquedo com três ondas.

  (1) onda S não-ressonante, real (referência);
  (2) ressonância R em s_ππ com L = 1;
  (3) K*(892) em s_Kπ com L = 1 (onda de canal cruzado, omitida no ajuste em H2).

Fatores angulares de Zemach para L = 1 e largura constante (Blatt–Weisskopf e Γ(s) ficam para o M1).
Convenção CKM da Eq. 2 do plano: c2 = |c2| exp[i(δ2 ∓ Δφ)], com −Δφ para B− (carga = −1).
"""

import numpy as np

from .cinematica import s_kpi_linha

ESCALA = 10.0  # GeV², só para deixar |A_r| de ordem 1


def breit_wigner(s, m_r, gamma):
    return 1.0 / (m_r * m_r - s - 1j * m_r * gamma)


def zemach_pipi(s_kpi, s_pipi, m):
    """L = 1 para R → π+π− com espectador K: s(K π±) − s(K π∓). Massas iguais: sem termo de correção."""
    return (s_kpi_linha(s_kpi, s_pipi, m) - s_kpi) / ESCALA


def zemach_kpi(s_kpi, s_pipi, m):
    """L = 1 para K* → K π∓ com espectador π±: s(K π±) − s(π π) + (m_π² − m_K²)(M_B² − m_π²)/s_Kπ."""
    corr = (m.pi**2 - m.K**2) * (m.B**2 - m.pi**2) / s_kpi
    return (s_kpi_linha(s_kpi, s_pipi, m) - s_pipi + corr) / ESCALA


def matriz_amplitudes(x, m, cfg):
    """Matriz N×3 complexa A_r(x_i), pré-calculada uma vez por amostra."""
    s_kpi, s_pipi = x[:, 0], x[:, 1]
    r, k = cfg["ressonancia_pipi"], cfg["kstar"]
    a1 = np.ones(len(x), dtype=complex)
    a2 = breit_wigner(s_pipi, r["m"], r["gamma"]) * zemach_pipi(s_kpi, s_pipi, m)
    a3 = breit_wigner(s_kpi, k["m"], k["gamma"]) * zemach_kpi(s_kpi, s_pipi, m)
    return np.column_stack([a1, a2, a3])


def coeficientes(theta, carga):
    """theta = (|c2|, δ2, |c3|, δ3, Δφ); carga = −1 (B−) ou +1 (B+)."""
    c2, d2, c3, d3, dphi = theta
    return np.array([1.0, c2 * np.exp(1j * (d2 + carga * dphi)), c3 * np.exp(1j * d3)])


def intensidade(amps, theta, carga):
    return np.abs(amps @ coeficientes(theta, carga)) ** 2


def assimetria_fisica(amps, theta):
    """Eq. 1: (|M−|² − |M+|²)/(|M−|² + |M+|²)."""
    im, ip = intensidade(amps, theta, -1), intensidade(amps, theta, +1)
    return (im - ip) / (im + ip)
