from scipy.fft import fft2, fftshift
from PIC import calculer_densite_pic_ordre1
import numpy as np

def obtenir_spectre_puissance(pos_agents, L=1.0, nb_cases=128):
    h = L / nb_cases #size of a case
    densite = calculer_densite_pic_ordre1(pos_agents, L, nb_cases, h)

    #make the 2D fourier transform
    fourier = fft2(densite)

    #give the spectrum (shift to have it in the center)
    spectre = np.abs(fftshift(fourier)) ** 2

    #Normalisation to compare to other spectrum
    spectre = spectre / np.sum(spectre)

    return spectre

#then the distance between two patterns is only taking the L2 discrete norm between their spectrum