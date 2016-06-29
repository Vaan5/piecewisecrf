import numpy as np

import piecewisecrf.tests.mean_field_test as test


def _exp_norm(marginals):
    '''

    Normalize marginals

    Parameters
    ----------
    marginals : numpy array with dimensions [height, width, number_of_classes]
        Input array

    Returns
    -------
    marginals : numpy array
        Normalized marginals

    '''
    # apply exponential function
    marginal_max = np.amax(marginals, axis=2)
    marginal_max = np.repeat(marginal_max, marginals.shape[2])
    marginal_max = np.reshape(marginal_max, marginals.shape)
    marginals = marginals - marginal_max
    marginals = np.exp(marginals)

    # normalize each marginal
    marginal_sum = np.sum(marginals, axis=2)
    marginal_sum = np.repeat(marginal_sum, marginals.shape[2])
    marginal_sum = np.reshape(marginal_sum, marginals.shape)

    marginals = marginals / marginal_sum

    return marginals


def mean_field(unary, pairwise, number_of_iterations=3, calculate_energy=False):
    '''

    Computes given number of mean field iterations

    Parameters
    ----------
    unary : numpy array with dimensions [height, width, number_of_classes]
        Unary scores (outputs from network). This method will negate them in order
        to get potentials

    pairwise: list of tuples (pairwise_scores, zipped_indices, decoding)
        Pairwise scores and appropriate methods to use them:
            pairwise_scores: pairwise net outputs (not yet potentials)
            zipped_indices: (first_, secod_index) list
            decoding: mapping index -> (first_, second_class)

    number_of_iterations: int
        Number of mean field iterations

    calculate_energy: bool
        Print out mf energy functional. (Used for testing)
        The energy functional needs grow from iteration to iteration

    Returns
    -------
    marginals : numpy array
        Normalized marginals


    '''
    width = unary.shape[1]
    number_of_classes = unary.shape[2]

    # multiple net outputs by -1 because they represent potentials
    unary = -unary

    # prepare pairwise potentials in matrix format
    pairwise3d = []
    for p, zipped_indices, decoding in pairwise:
        p = -1.0 * p
        pairwise3d.append(p.reshape([-1, number_of_classes, number_of_classes]))

    # initialize marginals to unary potentials
    marginals = np.zeros(unary.shape)
    np.copyto(marginals, -1.0 * unary)
    marginals = marginals.astype(np.float128)
    marginals = _exp_norm(marginals)

    if calculate_energy:
        pairwise_list = [(-1.0 * pair.reshape([-1, number_of_classes ** 2]), zipped_indices, decoding)
                         for pair, zipped_indices, decoding in pairwise]
        print(test.calculcate_energy(unary, pairwise_list, marginals))

    for it_num in range(number_of_iterations):
        # print("Mean-Field iteration #{}".format(it_num + 1))
        tmp_marginals = np.zeros(marginals.shape)
        np.copyto(tmp_marginals, -1.0 * unary)
        tmp_marginals = tmp_marginals.astype(np.float128)
        for ind, (p, zipped_indices, decoding) in enumerate(pairwise):
            for i, (f, s) in enumerate(zipped_indices):
                tmp_marginals[f // width, f % width, :] -= pairwise3d[ind][i].dot(marginals[s // width, s % width, :])
                tmp_marginals[s // width, s % width, :] -= marginals[f // width, f % width, :].dot(pairwise3d[ind][i])

        tmp_marginals = _exp_norm(tmp_marginals)
        np.copyto(marginals, tmp_marginals)
        marginals = marginals.astype(np.float128)

        if calculate_energy:
            pairwise_list = [(-1.0 * pair.reshape([-1, number_of_classes ** 2]), zipped_indices, decoding)
                             for pair, zipped_indices, decoding in pairwise]
            print(test.calculcate_energy(unary, pairwise_list, marginals))

    return marginals
