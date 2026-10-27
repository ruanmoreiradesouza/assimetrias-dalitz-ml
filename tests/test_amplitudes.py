"""Testes analíticos do modelo (critério de pronto do M1 antecipado no M0)."""

import math

import numpy as np

from dalitzcp.amplitudes import (
    ESCALA,
    assimetria_fisica,
    breit_wigner,
    intensidade,
    matriz_amplitudes,
    zemach_kpi,
    zemach_pipi,
)
from dalitzcp.cinematica import espaco_fase, kallen, s_kpi_linha
from dalitzcp.sementes import gerador


def amostra(massas, n=20_000, semente=10):
    return espaco_fase(gerador(semente, "testes"), n, massas)


def test_dphi_zero_implica_assimetria_nula(massas, cfg, theta):
    amps = matriz_amplitudes(amostra(massas), massas, cfg)
    t = list(theta)
    t[4] = 0.0
    assert np.max(np.abs(assimetria_fisica(amps, t))) < 1e-12


def test_identidade_da_interferencia_eq2(massas, cfg):
    # duas ondas (|c3| = 0): |M−|² − |M+|² = 4|c1||c2| A1 K2 sin Δφ [sin Δδ0 Re BW + cos Δδ0 Im BW]
    x = amostra(massas)
    amps = matriz_amplitudes(x, massas, cfg)
    r = cfg["ressonancia_pipi"]
    bw = breit_wigner(x[:, 1], r["m"], r["gamma"])
    k2 = zemach_pipi(x[:, 0], x[:, 1], massas)
    for dd0 in (0.0, math.pi / 2, 0.7, -2.1):
        t = [0.8, dd0, 0.0, 0.0, 0.3]
        dif = intensidade(amps, t, -1) - intensidade(amps, t, +1)
        prev = 4 * 0.8 * k2 * math.sin(0.3) * (math.sin(dd0) * bw.real + math.cos(dd0) * bw.imag)
        assert np.max(np.abs(dif - prev)) / np.max(np.abs(prev)) < 1e-12


def test_troca_de_sinal_em_mR2_para_dd0_pi_sobre_2(massas, cfg):
    # com Δδ0 = π/2 a assimetria (dividida pelo fator angular) troca de sinal exatamente em s_ππ = m_R²
    m_r = cfg["ressonancia_pipi"]["m"]
    s_pp = np.array([m_r**2 - 0.05, m_r**2 + 0.05])
    x = np.column_stack([np.full(2, 6.0), s_pp])
    amps = matriz_amplitudes(x, massas, cfg)
    t = [0.8, math.pi / 2, 0.0, 0.0, 0.3]
    dif = intensidade(amps, t, -1) - intensidade(amps, t, +1)
    sinal = np.sign(dif / zemach_pipi(x[:, 0], x[:, 1], massas))
    assert sinal[0] == -sinal[1]


def test_simetria_cp_troca_cargas(massas, cfg, theta):
    # trocar Δφ → −Δφ é o mesmo que trocar B− ↔ B+
    amps = matriz_amplitudes(amostra(massas, 2000), massas, cfg)
    t_inv = list(theta)
    t_inv[4] = -theta[4]
    assert np.allclose(intensidade(amps, theta, -1), intensidade(amps, t_inv, +1), rtol=1e-13)


def test_fatores_de_zemach_sao_4pq_cos_theta_helicidade(massas):
    # referência independente: no repouso do par (a b), Z_L=1 = ±4|p||q| cos θ_H / ESCALA,
    # com q o momento de a e p o do espectador c; cos θ_H vem de s(a c)
    x = amostra(massas, 5000, semente=11)
    s_kp, s_pp = x[:, 0], x[:, 1]
    s_lin = s_kpi_linha(s_kp, s_pp, massas)
    M, mk, mp = massas.B, massas.K, massas.pi

    def z_ref(s_ab, m_a, m_b, m_c, s_ac):
        raiz = np.sqrt(s_ab)
        e_a = (s_ab + m_a**2 - m_b**2) / (2 * raiz)
        e_c = (M**2 - s_ab - m_c**2) / (2 * raiz)
        q = np.sqrt(kallen(s_ab, m_a**2, m_b**2)) / (2 * raiz)
        p = np.sqrt(kallen(M**2, s_ab, m_c**2)) / (2 * raiz)
        cos = (m_a**2 + m_c**2 + 2 * e_a * e_c - s_ac) / (2 * p * q)
        return 4 * p * q * cos / ESCALA

    # K* → K π∓ (a = K, b = π∓), espectador π±: s(a c) = s(K π±)
    razao = zemach_kpi(s_kp, s_pp, massas) / z_ref(s_kp, mk, mp, mp, s_lin)
    assert np.allclose(np.abs(razao), 1.0, rtol=1e-9) and np.all(np.sign(razao) == np.sign(razao[0]))
    # R → π∓ π± (a = π∓, b = π±), espectador K: s(a c) = s(K π∓)
    razao = zemach_pipi(s_kp, s_pp, massas) / z_ref(s_pp, mp, mp, mk, s_kp)
    assert np.allclose(np.abs(razao), 1.0, rtol=1e-9) and np.all(np.sign(razao) == np.sign(razao[0]))
