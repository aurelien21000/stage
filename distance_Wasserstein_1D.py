from scipy.stats import wasserstein_distance

from PIC import calculer_densite_pic_ordre1



def distance_wasserstein_1D(pos_agent1,pos_agent2, L_ESPACE, nb_cases):

    densite1 = calculer_densite_pic_ordre1(pos_agent1, L_ESPACE, nb_cases)
    densite2 = calculer_densite_pic_ordre1(pos_agent2, L_ESPACE, nb_cases)

    dist_wasserstein = wasserstein_distance(densite1.ravel(), densite2.ravel())

    return dist_wasserstein

#choosing the grid is made when we discretise the density in "calculer_densite_pic_ordre1"
#.ravel make the 2D density in a 1D array and wasserstein distance turn those arrays into signature
# and calculate the wasserstein distance between the histograms (continuous so that it doesn't depend
#on the number of bins)