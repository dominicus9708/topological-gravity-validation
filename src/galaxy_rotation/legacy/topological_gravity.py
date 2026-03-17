"""
Topological gravity equations
"""

from constants import c_info, xi


def compute_phi_topo(sigma):
    """
    Φ_topo = ξ c_info^2 σ
    """

    phi_topo = xi * (c_info ** 2) * sigma

    return phi_topo


def compute_g_eff(g_N, sigma_gradient):
    """
    g_eff = g_N − ξ c_info^2 ∇σ
    """

    g_eff = g_N - xi * (c_info ** 2) * sigma_gradient

    return g_eff
