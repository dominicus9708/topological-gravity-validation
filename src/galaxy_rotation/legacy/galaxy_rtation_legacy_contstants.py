# src/constants.py

from __future__ import annotations

# ----- SI constants -----

G_SI = 6.67430e-11          # m^3 / (kg s^2)
C_SI = 2.99792458e8         # m / s

# ----- astrophysical constants -----

# Gravitational constant in astrophysical units:
# (km/s)^2 * kpc / Msun
G_ASTRO = 4.30091e-6

MSUN_IN_KG = 1.98847e30
KPC_IN_M = 3.085677581491367e19
PC_IN_M = 3.085677581491367e16
MPC_IN_M = 3.085677581491367e22

# ----- project baseline parameters -----

# background structural degree
D_BG = 2.0

# coupling / scaling parameters for the current paper-4 pipeline
SIGMA_ALPHA = 1.0
SIGMA_EPSILON = 1.0e-12

# information-speed parameter
# keep both explicit until final theoretical normalization is fixed
C_INFO_SI = C_SI
C_INFO_NORM = 3.0
