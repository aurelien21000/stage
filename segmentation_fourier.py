import numpy as np


def filtrer_spectre_pour_ellipse(spectrum, cutoff=1.3e-3):
    nb_cases = spectrum.shape[0]
    center = nb_cases // 2
    

    spectrum_clean = np.copy(spectrum)
    spectrum_clean[center, center] = 0.0 #remove the continuous component

    i_max = np.max(spectrum_clean)
    if i_max < cutoff: #uniform state so we remove everything (and return false to simplify functions after)
        return False, np.zeros_like(spectrum), ([], [], [])

    spectrum_filtered = np.where(spectrum_clean >= cutoff, spectrum_clean, 0.0)
    lines, columns = np.nonzero(spectrum_filtered) #keeping only the pixels over the cut-off
    
    Y = lines - center
    X = columns - center  #to center the (X,Y) axis for drawing after
    Wheights = spectrum_filtered[lines, columns] #keep the original intensity for the others

    return True, spectrum_filtered, (X, Y, Wheights)


def calculer_parametres_ellipse(coords):
    X, Y, Weights = coords

    if len(X) < 2:
        return 0.0, 0.0, 0.0, 0.0 #fourier spectrum is symmetrical

    W = np.sum(Weights)
    if W == 0:
        return 0.0, 0.0, 0.0, 0.0

    x_bar = np.sum(X * Weights) / W
    y_bar = np.sum(Y * Weights) / W #moments of order 1

    mu_xx = np.sum(Weights * (X - x_bar) ** 2) / W
    mu_yy = np.sum(Weights * (Y - y_bar) ** 2) / W
    mu_xy = np.sum(Weights * (X - x_bar) * (Y - y_bar)) / W #Covariance matrice in the lateX

    C = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])
    eigen_values, eigen_vector = np.linalg.eigh(C)

    lambda_min = eigen_values[0]
    lambda_max = eigen_values[1] #linalg is in this order

    factor = 3.0 #choice of the factor in front of the std (3 is taking almost all the total values can be reduce to 2)
    semi_major_axis = factor * np.sqrt(lambda_max)
    semi_minor_axis = factor * np.sqrt(lambda_min)

    excentricity = np.sqrt(max(0.0, 1.0 - (lambda_min / lambda_max)))#def in the lateX

    v_max = eigen_vector[:, 1]
    angle = np.arctan2(v_max[1], v_max[0])

    return semi_major_axis, semi_minor_axis, excentricity, angle