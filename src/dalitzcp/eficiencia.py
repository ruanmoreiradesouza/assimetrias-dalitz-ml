"""Eficiência ε±(x) = (1 ± A_det) ε0(x), com ε0 conhecido e linear em s_Kπ (hipótese explícita do plano)."""

from .cinematica import faixa_s_kpi


def eficiencia(x, carga, m, cfg):
    e = cfg["eficiencia"]
    hi = faixa_s_kpi(m)[1]
    return (1 + carga * cfg["A_det"]) * (e["e0"] + e["e1"] * x[:, 0] / hi)
