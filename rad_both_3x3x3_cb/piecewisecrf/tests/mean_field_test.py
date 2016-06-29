import numpy as np


def calculcate_energy(unary, pairwise, marginals):
    '''

    Calculates energy for the given unary and pairwise potentials.
    Used for testing whether mean field implementation is correct or not.

    Parameters
    ----------
    unary : numpy array with dimensions [height, width, number_of_classes]
            Unary potentials

    pairwise: list of tuples (pairwise_potentials, (first_, second_index), decoding)
            pairwise_potentials - array with dimensions [nubmer_of_neighbours, number_of_classes ** 2]
            decoding - decoding dictionary which maps index -> label pair
            (first_, second_index) - pixel index pair in pairwise potentials tensor


    marginals: numpy array with dimensions [height, width, number_of_classes]
            Marginal distributions computed with mean field approximation algorithm

    Returns
    -------
    energy: float
            Energy function value for given unary and pairwise potentials


    '''

    height = unary.shape[0]
    width = unary.shape[1]
    number_of_classes = unary.shape[2]

    assert height == marginals.shape[0]
    assert width == marginals.shape[1]
    assert number_of_classes == marginals.shape[2]

    energy = 0.0
    # unary potentials contribution
    for h in range(height):
        for w in range(width):
            marginals_i = marginals[h, w, :]
            unary_i = unary[h, w, :]
            energy -= np.sum(marginals_i * unary_i)

    # pairwise potentials contribution
    pairwise_energy_contribution = 0.0
    for p, indices, decoding in pairwise:
        for index, (f, s) in enumerate(indices):
            marginals_f = marginals[f // width, f % width, :]
            marginals_s = marginals[s // width, s % width, :]

            pairwise_f_s = p[index, :]

            q_yf_ys = np.zeros(pairwise_f_s.shape)

            for d, (yf, ys) in decoding.items():
                q_yf_ys[d] = marginals_f[yf] * marginals_s[ys]

            pairwise_energy_contribution += np.sum(q_yf_ys * pairwise_f_s)

    energy -= pairwise_energy_contribution

    # entropy contribution
    for h in range(height):
        for w in range(width):
            q_i = marginals[h, w, :]
            energy -= np.sum(q_i * np.log(q_i).clip(min=1e-10))

    return energy
