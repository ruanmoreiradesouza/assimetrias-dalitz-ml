"""Ajuste: Φ hermitiana, recuperação dos parâmetros e concordância com o iminuit (critério do M2)."""

import numpy as np
import pytest

from dalitzcp.geracao import gerar_amostra
from dalitzcp.sementes import gerador
from dalitzcp.verossimilhanca import NOMES, ajustar, ajustar_minuit, amplitudes_da_amostra, matriz_phi, q_cp


@pytest.fixture(scope="module")
def phis(massas, cfg):
    return matriz_phi(gerador(cfg["semente_mae"], "normalizacao"), massas, cfg)


@pytest.fixture(scope="module")
def amps(massas, cfg, theta):
    amostra = gerar_amostra(gerador(cfg["semente_mae"], "testes", 1), theta, cfg["N_eventos"], massas, cfg)
    return amplitudes_da_amostra(amostra, massas, cfg)


def test_phi_hermitiana_e_positiva(phis):
    for p in phis.values():
        assert np.allclose(p, p.conj().T, atol=1e-15)
        assert np.all(np.linalg.eigvalsh(p) > 0)


def test_adet_so_reescala_phi(phis, cfg):
    a = cfg["A_det"]
    assert np.allclose(phis[-1] / (1 - a), phis[+1] / (1 + a), rtol=1e-12)


def test_ajuste_recupera_theta_e_concorda_com_iminuit(amps, phis, theta, cfg):
    r = ajustar(amps, phis, theta, gerador(cfg["semente_mae"], "testes", 2), partidas=1)
    m = ajustar_minuit(amps, phis, theta)
    assert m.valid
    erros = np.array(m.errors)
    ajustado = dict(zip(NOMES, m.values, strict=True))
    assert np.all(np.abs(np.array(m.values) - np.array(theta)) < 5 * erros), ajustado
    assert np.max(np.abs(r.x - np.array(m.values)) / erros) < 1e-2


def test_q_cp_nao_negativo_e_detecta_dphi(amps, phis, theta, cfg):
    q, livre, nulo = q_cp(amps, phis, theta, gerador(cfg["semente_mae"], "testes", 3))
    assert q >= 0 and nulo.x[4] == 0.0
    assert q > 25  # Δφ = 0,25 com N = 5000 dá Δφ̂/σ ≈ 8 no piloto
