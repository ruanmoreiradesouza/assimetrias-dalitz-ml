"""Estatísticas de forma do plano: S_j ponderado de Miranda (Eq. 5) e conversão comum para A_fís (Eq. 6).

A grade adaptativa (k-d tree), S_CP com células ocupadas e a estatística T_ML entram no M3.
"""

import numpy as np


def contagens_ponderadas(x, w, bordas_kpi, bordas_pipi):
    """W_j = Σ w_i e V_j = Σ w_i² numa grade retangular (células fora do Dalitz ficam com zero)."""
    W, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[bordas_kpi, bordas_pipi], weights=w)
    V, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[bordas_kpi, bordas_pipi], weights=w * w)
    return W.ravel(), V.ravel()


def miranda_ponderado(Wm, Vm, Wp, Vp):
    """S_j = (W_j− − α W_j+)/√(V_j− + α² V_j+), α = W−/W+. Devolve S_j só das células ocupadas."""
    alfa = Wm.sum() / Wp.sum()
    var = Vm + alfa**2 * Vp
    ocupadas = var > 0
    return (Wm[ocupadas] - alfa * Wp[ocupadas]) / np.sqrt(var[ocupadas])


def para_fisica(a_bal, g):
    """Eq. 6: Â_fís = (Â_bal + g)/(1 + g Â_bal), com g = (W− − W+)/(W− + W+) (soma de tanh)."""
    return (a_bal + g) / (1 + g * a_bal)
