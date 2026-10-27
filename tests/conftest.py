import pytest

from dalitzcp.cinematica import Massas
from dalitzcp.config import carregar_config, theta_nominal


@pytest.fixture(scope="session")
def cfg():
    return carregar_config()


@pytest.fixture(scope="session")
def massas(cfg):
    return Massas.de_config(cfg)


@pytest.fixture(scope="session")
def theta(cfg):
    return theta_nominal(cfg)
