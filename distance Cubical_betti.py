import numpy as np
import gudhi
from gudhi.representations import BettiCurve

#We suppose we already discretise the density on a grid using pic method (function showed in another file)

def extraire_betti_cubique_periodique(densite, resolution=100):
    #We normalise each density so maximum is one
    max_densite = np.max(densite)
    if max_densite > 0:
        densite_norm = densite / max_densite
    else:
        densite_norm = densite

    #inverse because cubical complex start at the minimum
    densite_inverse = -densite_norm

    complexe = gudhi.PeriodicCubicalComplex(
        top_dimensional_cells=densite_inverse,
        periodic_dimensions=[True, True]
    )
    complexe.compute_persistence()
    diag_0 = complexe.persistence_intervals_in_dimension(0)

    #last connexe component is living forever so for betti curve we have to replace inf by 0.
    if len(diag_0) == 0:
        diag_0_fini = np.empty((0, 2))
    else:
        diag_0_fini = np.array([[b, d if d != float('inf') else 0.0] for b, d in diag_0])

    #evaluating betti curve for the same grid for each datas to justify L2 norm
    bc = BettiCurve(resolution=resolution, sample_range=[-1.0, 0.0])
    return bc.fit_transform([diag_0_fini])[0]

#For the distance we are just taking the L2 discrete norm (sum from 1 to resolution) on the points in betticurve