import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from tqdm import tqdm
from numba import njit


@njit(nogil=True)
def calculer_densite_pic_ordre1(pos_agents, L, nb_cases, h):
    M = pos_agents.shape[0]
    grille_noeuds = np.zeros((nb_cases, nb_cases), dtype=np.float64)

    for k in range(M):
        x = pos_agents[k, 0] % L
        y = pos_agents[k, 1] % L
        x_norm = x / h
        y_norm = y / h
        i = int(np.floor(x_norm))
        j = int(np.floor(y_norm))
        dx = x_norm - i
        dy = y_norm - j
        i0, j0 = i % nb_cases, j % nb_cases
        i1, j1 = (i + 1) % nb_cases, (j + 1) % nb_cases

        grille_noeuds[i0, j0] += (1.0 - dx) * (1.0 - dy)
        grille_noeuds[i1, j0] += dx * (1.0 - dy)
        grille_noeuds[i0, j1] += (1.0 - dx) * dy
        grille_noeuds[i1, j1] += dx * dy

    densite_cellules = np.zeros((nb_cases, nb_cases), dtype=np.float64)
    for i in range(nb_cases):
        for j in range(nb_cases):
            i1, j1 = (i + 1) % nb_cases, (j + 1) % nb_cases
            densite_cellules[i, j] = 0.25 * (
                    grille_noeuds[i, j] + grille_noeuds[i1, j] +
                    grille_noeuds[i, j1] + grille_noeuds[i1, j1]
            )
    return densite_cellules / (M * (h ** 2))