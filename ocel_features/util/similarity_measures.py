import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import wasserstein_distance


def compute_cosine_similarity_matrix(df):
    return cosine_similarity(df.values)


def compute_histo_wasserstein_similarity(df):
    data_matrix = normalize(df.values)
    sim_matrix = np.zeros((data_matrix.shape[0], data_matrix.shape[0]))

    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[0]):
            if i == j:
                sim_matrix[i][j] = 1.0
            elif sim_matrix[i][j] == 0.0:
                sim = 1 - wasserstein_distance(data_matrix[i], data_matrix[j])
                sim_matrix[i][j] = sim_matrix[j][i] = sim

    return sim_matrix
