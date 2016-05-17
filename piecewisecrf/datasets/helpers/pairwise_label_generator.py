import numpy as np
import piecewisecrf.datasets.cityscapes.prefs as prefs

FLAGS = prefs.flags.FLAGS


def generate_encoding_decoding_dict(number_of_labels):
    '''

    Generates mappings between label pairs and indices

    Parameters
    ----------
    number_of_labels : int
        Number of classes in the dataset

    Returns
    -------
    encoding, decoding: tuple
        encoding is a dict that maps (label, label) => index
        decoding is a dict that maps index => (label, label)


    '''
    decoding = dict(enumerate((i, j) for i in range(number_of_labels) for j in range(number_of_labels)))
    encoding = {v: k for k, v in list(decoding.items())}
    return encoding, decoding


def generate_pairwise_labels(labels, indices_getter, number_of_labels):
    '''

    Generates ground truth map for pairwise potentials. The neighbourhood is
    defined via the indices getter callable.

    Parameters
    ----------
    labels: numpy array
        Ground truth map for unary potentials (label image from the dataset)

    indices_getter: callable
        Function that returns two lists with indices denoting neighbour pixels

    number_of_labels : int
        Number of classes in the dataset

    Returns
    -------
    pairwise_labels: numpy array
        Ground truth map for pairwise potentials


    '''
    flattened_labels = np.reshape(labels, [-1])
    first_index, second_index = indices_getter()
    index_pairs = list(zip(first_index, second_index))

    encoding, decoding = generate_encoding_decoding_dict(number_of_labels)

    pairwise_labels = np.array([encoding.get((flattened_labels[x[0]],
                                              flattened_labels[x[1]]), -1) for x in index_pairs])
    return pairwise_labels.astype(np.int32)


#######################################################################################################################
####                             Pairwise potentials modelling above/below relations                               ####
#######################################################################################################################


def get_indices_surrounding():
    '''

    Returns two lists with pixel indices for surrounding pixels

    Returns
    -------
    original_container, container: lists
        original_container contains indices for the first pixel in the neighbourhood
        container contains indices for the second pixel in the neighbourhood


    '''
    container = []
    original_container = []
    h = int(FLAGS.img_height / FLAGS.subsample_factor)
    w = int(FLAGS.img_width / FLAGS.subsample_factor)
    nsize = FLAGS.surrounding_neighbourhood_size
    nsize_half = int(nsize / 2)
    for i in range(h):
        for j in range(w):
            index_1d = i * w + j
            for n_i in range(i - nsize_half, i + nsize_half + 1):
                for n_j in range(j - nsize_half, j + nsize_half + 1):
                    if n_i < 0 or n_i >= h or n_j < 0 or n_j >= w or (n_i == i and n_j == j):
                        continue
                    container.append(n_i * w + n_j)
                    original_container.append(index_1d)
    return original_container, container


def get_number_of_all_neigbhours_surrounding(h, w, nsize):
    '''

    Returns total number of neighbours for the surrounding neighbourhood

    Returns
    -------
    ret_val: int
        Total number of neighbours


    '''
    ret_val = 0
    nsize_half = int(nsize / 2)
    for i in range(int(h)):
        for j in range(int(w)):
            for n_i in range(i - nsize_half, i + nsize_half + 1):
                for n_j in range(j - nsize_half, j + nsize_half + 1):
                    if n_i < 0 or n_i >= h or n_j < 0 or n_j >= w or (n_i == i and n_j == j):
                        continue
                    ret_val += 1
    return ret_val


FIRST_INDICES_SURR, SECOND_INDICES_SURR = get_indices_surrounding()

NUMBER_OF_NEIGHBOURS_SURR = get_number_of_all_neigbhours_surrounding(
    FLAGS.img_height / FLAGS.subsample_factor,
    FLAGS.img_width / FLAGS.subsample_factor,
    FLAGS.surrounding_neighbourhood_size
)


#############################################################
###  Pairwise potentials modelling above/below relations  ###
#############################################################


def get_indices_above_below():
    '''

    Returns two lists with pixel indices for above/below pixels

    Returns
    -------
    original_container, container: lists
        original_container contains indices for the first pixel in the neighbourhood
        container contains indices for the second pixel in the neighbourhood


    '''
    container = []
    original_container = []
    h = 19#int(FLAGS.img_height / FLAGS.subsample_factor)
    w = 38#int(FLAGS.img_width / FLAGS.subsample_factor)
    nsize_width = 3#FLAGS.neigbourhood_above_below_width
    nsize_height = 4#FLAGS.neigbourhood_above_below_height
    nsize_width_half = int(nsize_width / 2)
    for i in range(h):
        for j in range(w):
            index_1d = i * w + j
            for n_i in range(i - nsize_height + 1, i):
                for n_j in range(j - nsize_width_half, j + nsize_width_half + 1):
                    if n_i < 0 or n_i >= h or n_j < 0 or n_j >= w or (n_i == i and n_j == j):
                        continue
                    container.append(n_i * w + n_j)
                    original_container.append(index_1d)
    return original_container, container


def get_number_of_all_neigbhours_above_below(h, w, nsize_height, nsize_width):
    '''

    Returns total number of neighbours for the above/below neighbourhood

    Returns
    -------
    ret_val: int
        Total number of neighbours


    '''
    ret_val = 0
    nsize_width_half = int(nsize_width / 2)
    for i in range(int(h)):
        for j in range(int(w)):
            for n_i in range(i - nsize_height + 1, i):
                for n_j in range(j - nsize_width_half, j + nsize_width_half + 1):
                    if n_i < 0 or n_i >= h or n_j < 0 or n_j >= w or (n_i == i and n_j == j):
                        continue
                    ret_val += 1
    return ret_val


FIRST_INDICES_AB, SECOND_INDICES_AB = get_indices_above_below()

NUMBER_OF_NEIGHBOURS_AB = get_number_of_all_neigbhours_above_below(
    FLAGS.img_height / FLAGS.subsample_factor,
    FLAGS.img_width / FLAGS.subsample_factor,
    FLAGS.neigbourhood_above_below_height,
    FLAGS.neigbourhood_above_below_width
)
