import numpy as np


def calculate_weights(labels, num_classes):
    '''

    Calculate class balancing weights for unary potentials

    Parameters
    ----------
    labels: numpy array
        Array with ground truth labels

    num_classes: numpy array
        Number of classes in the dataset

    Returns
    -------
    label_weights: numpy array
        Image of weights

    weights: numpy array
        Per class weights


    '''
    height = labels.shape[0]
    width = labels.shape[1]

    class_hist = np.zeros(num_classes, dtype=np.int64)
    for i in range(height):
        for j in range(width):

            if labels[i, j] >= 0 and labels[i, j] < num_classes:
                class_hist[labels[i, j]] += 1

    num_labels = class_hist.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    max_wgt = 1000
    for i in range(num_classes):
        if class_hist[i] > 0:
            weights[i] = min(max_wgt, 1.0 / (class_hist[i] / num_labels))
        else:
            weights[i] = 0
    label_weights = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            cidx = labels[i, j]
            if cidx >= 0 and cidx < num_classes:
                label_weights[i, j] = weights[cidx]
    return label_weights, weights


def calculate_weights_binary(weights, labels_pairwise, decoding, num_classes):
    '''

    Calculate class balancing weights for binary potentials

    Parameters
    ----------
    weights: numpy array
        Per class weights for unary potentials

    labels_pairwise: numpy array
        Array with ground truth labels for binary potentials

    decoding: dict
        dict that maps index => (label, label)

    num_classes: numpy array
        Number of classes in the dataset

    Returns
    -------
    ret: numpy array
        Image of weights


    '''
    ret = np.zeros(labels_pairwise.shape, dtype=np.float32)
    for i in range(ret.shape[0]):
        cidx1, cidx2 = decoding.get(labels_pairwise[i], (-1, -1))
        if cidx1 >= 0 and cidx1 < num_classes and cidx2 >= 0 and cidx2 < num_classes:
            ret[i] = min(100, weights[cidx1] * weights[cidx2])
    return ret
