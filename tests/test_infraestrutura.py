import numpy as np
import pytest

from dalitzcp.config import RAIZ, carregar_config
from dalitzcp.sementes import ETAPAS, gerador


def test_config_independe_do_diretorio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = carregar_config()
    assert cfg["N_eventos"] == 5000
    assert (RAIZ / "pyproject.toml").exists()


def test_mesmo_toy_mesmos_numeros():
    a = gerador(1, "H1", 7).normal(size=5)
    b = gerador(1, "H1", 7).normal(size=5)
    assert np.array_equal(a, b)


def test_fluxos_distintos_entre_toys_e_etapas():
    base = gerador(1, "H1", 7).normal(size=5)
    assert not np.array_equal(base, gerador(1, "H1", 8).normal(size=5))
    assert not np.array_equal(base, gerador(1, "H2", 7).normal(size=5))
    assert not np.array_equal(base, gerador(2, "H1", 7).normal(size=5))


def test_etapas_tem_indices_unicos():
    assert len(set(ETAPAS.values())) == len(ETAPAS)


def test_etapa_desconhecida_falha():
    with pytest.raises(KeyError):
        gerador(1, "H9")
