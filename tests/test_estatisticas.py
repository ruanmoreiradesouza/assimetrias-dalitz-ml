import numpy as np

from dalitzcp.estatisticas import contagens_ponderadas, miranda_ponderado, para_fisica


def test_eq6_e_soma_de_tanh():
    rng = np.random.default_rng(0)
    pm, pp = rng.uniform(0.1, 2, 50), rng.uniform(0.1, 2, 50)
    r = 1.07
    a_bal = (pm - pp) / (pm + pp)
    g = (r - 1) / (r + 1)
    a_fis = (r * pm - pp) / (r * pm + pp)
    assert np.max(np.abs(para_fisica(a_bal, g) - a_fis)) < 1e-14


def test_miranda_nulo_quando_forma_igual():
    # B− = 2 × B+ célula a célula: só assimetria global, que α desconta
    wp = np.array([10.0, 20.0, 5.0, 0.0])
    s = miranda_ponderado(2 * wp, 2 * wp, wp, wp)
    assert len(s) == 3 and np.allclose(s, 0)


def test_miranda_sem_pesos_coincide_com_bediaga():
    # com pesos 1: V = N e S_j = (N− − αN+)/√(N− + α²N+)
    nm, np_ = np.array([30.0, 50.0]), np.array([40.0, 45.0])
    alfa = nm.sum() / np_.sum()
    s = miranda_ponderado(nm, nm, np_, np_)
    assert np.allclose(s, (nm - alfa * np_) / np.sqrt(nm + alfa**2 * np_))


def test_contagens_ponderadas():
    x = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    w = np.array([2.0, 3.0, 4.0])
    W, V = contagens_ponderadas(x, w, [0, 1, 2], [0, 1, 2])
    assert np.allclose(W, [2, 3, 0, 4]) and np.allclose(V, [4, 9, 0, 16])
