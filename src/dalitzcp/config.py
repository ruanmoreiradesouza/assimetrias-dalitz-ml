"""Leitura do config.json, com caminhos relativos à raiz do repositório (não ao diretório de execução)."""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


def carregar_config(caminho=None):
    caminho = Path(caminho) if caminho is not None else RAIZ / "config.json"
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def theta_nominal(cfg):
    t = cfg["theta_nominal"]
    return [t["c2"], t["delta2"], t["c3"], t["delta3"], t["dphi"]]
