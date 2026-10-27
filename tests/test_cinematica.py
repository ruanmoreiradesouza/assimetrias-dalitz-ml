"""A cinemática é conferida por uma referência independente: reconstruo os quadrivetores no repouso do B
a partir de (s_Kπ, s_ππ) e verifico as massas e os invariantes."""

import numpy as np

from dalitzcp.cinematica import dentro, espaco_fase, kallen, limites_s_kpi, s_kpi_linha
from dalitzcp.sementes import gerador


def quadrivetores(s_kpi, s_pipi, m):
    """K (π∓) (π±) no repouso do B: K no eixo z, π± no plano xz, π∓ fecha o momento."""
    s_lin = s_kpi_linha(s_kpi, s_pipi, m)
    e_k = (m.B**2 + m.K**2 - s_pipi) / (2 * m.B)
    e_pl = (m.B**2 + m.pi**2 - s_kpi) / (2 * m.B)   # π± é o espectador do par K π∓
    p_k = np.sqrt(e_k**2 - m.K**2)
    p_pl = np.sqrt(e_pl**2 - m.pi**2)
    cos = (m.K**2 + m.pi**2 + 2 * e_k * e_pl - s_lin) / (2 * p_k * p_pl)
    sen = np.sqrt(np.clip(1 - cos**2, 0, None))
    k = np.stack([e_k, 0 * e_k, 0 * e_k, p_k], -1)
    pl = np.stack([e_pl, p_pl * sen, 0 * e_pl, p_pl * cos], -1)
    pm = np.array([m.B, 0, 0, 0]) - k - pl
    return k, pm, pl, cos


def massa2(p):
    return p[..., 0] ** 2 - (p[..., 1:] ** 2).sum(-1)


def test_quadrivetores_reproduzem_massas_e_invariantes(massas):
    x = espaco_fase(gerador(1, "testes"), 20_000, massas)
    k, pm, pl, cos = quadrivetores(x[:, 0], x[:, 1], massas)
    assert np.all(np.abs(cos) <= 1 + 1e-12)
    assert np.allclose(massa2(pm), massas.pi**2, rtol=0, atol=1e-9)
    assert np.allclose(massa2(k + pm), x[:, 0], rtol=1e-12, atol=1e-10)
    assert np.allclose(massa2(pl + pm), x[:, 1], rtol=1e-12, atol=1e-10)


def test_soma_das_invariantes(massas):
    x = espaco_fase(gerador(2, "testes"), 1000, massas)
    s3 = s_kpi_linha(x[:, 0], x[:, 1], massas)
    assert np.allclose(x[:, 0] + x[:, 1] + s3, massas.B**2 + massas.K**2 + 2 * massas.pi**2, rtol=1e-14)


def test_bordas_correspondem_a_cos_theta_igual_a_mais_ou_menos_um(massas):
    s_pp = np.linspace(0.1, 22.0, 50)
    lo, hi = limites_s_kpi(s_pp, massas)
    for borda, sinal in ((lo, 1.0), (hi, -1.0)):
        cos = quadrivetores(borda, s_pp, massas)[3]
        assert np.allclose(np.abs(cos), 1.0, atol=1e-7)
        assert np.all(dentro(borda * (1 + sinal * 1e-9), s_pp, massas))


def test_limites_pela_forma_de_kallen(massas):
    # s_Kπ extremos = m_K² + m_π² + 2(E_K E_π ∓ p_K p_π) no repouso do ππ, com p via Källén
    s = np.array([0.5, 4.0, 15.0])
    p_pi = np.sqrt(kallen(s, massas.pi**2, massas.pi**2)) / (2 * np.sqrt(s))
    p_k = np.sqrt(kallen(massas.B**2, s, massas.K**2)) / (2 * np.sqrt(s))
    e_pi, e_k = np.sqrt(p_pi**2 + massas.pi**2), np.sqrt(p_k**2 + massas.K**2)
    lo = massas.K**2 + massas.pi**2 + 2 * (e_k * e_pi - p_k * p_pi)
    hi = massas.K**2 + massas.pi**2 + 2 * (e_k * e_pi + p_k * p_pi)
    lo2, hi2 = limites_s_kpi(s, massas)
    assert np.allclose(lo, lo2, rtol=1e-12) and np.allclose(hi, hi2, rtol=1e-12)


def test_espaco_fase_uniforme(massas):
    # densidade uniforme: a fração em s_ππ < s0 bate com a área relativa (integração por trapézios)
    x = espaco_fase(gerador(3, "testes"), 200_000, massas)
    grade = np.linspace(4 * massas.pi**2, (massas.B - massas.K) ** 2, 20_001)
    lo, hi = limites_s_kpi(grade, massas)
    largura = hi - lo
    s0 = 5.0
    area = np.trapezoid(np.where(grade < s0, largura, 0), grade) / np.trapezoid(largura, grade)
    frac = np.mean(x[:, 1] < s0)
    assert abs(frac - area) < 5 * np.sqrt(area * (1 - area) / len(x))
