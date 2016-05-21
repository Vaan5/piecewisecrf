import numpy as np
import tensorflow as tf

from piecewisecrf.slim import slim
from piecewisecrf.slim import ops
from piecewisecrf.slim import scopes

import piecewisecrf.config.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as indices
import piecewisecrf.models.losses as losses

FLAGS = prefs.flags.FLAGS


def convolve(inputs, num_maps, k, name, init_layers=None, activation=tf.nn.relu):
    '''
    2D Convolution layer

    Parameters
    ----------
    inputs : tensorflow tensor
        Layer input tensor

    num_maps: int
        Number of output maps

    k: int
        Kernel size

    name: str
        Scope name

    init_layer: dict
        Map used for initializing layer weights

    activation: callable
        Activation function


    Returns
    -------
    op: tensorflow operation
        Convolution operation


    '''
    if init_layers:
        init_map = {'weights': init_layers[name + '/weights'],
                    'biases': init_layers[name + '/biases']}
    else:
        init_map = None
    return slim.ops.conv2d(inputs, num_maps, [k, k], scope=name, init=init_map, activation=activation)


def read_conv_params(in_dict, name):
    '''

    Helper function used for reading layer weights and biases

    Parameters
    ----------
    name: str
        Layer name

    in_dict: dict
        Map used for initializing layer weights


    Returns
    -------
    weights, biases: numpy arrays
        Layer weights and biases


    '''
    weights = in_dict[name][0]
    biases = in_dict[name][1]
    return weights, biases


def read_vgg_init(in_file_path):
    '''

    Read trained weights and biases from (numpy) binary file

    Parameters
    ----------
    in_file_path: str
        Input file path

    Returns
    -------
    layers: dict
        Layer weights and biases

    names: list
        List of layer names


    '''
    names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
             'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    layers = {}
    in_file = np.load(in_file_path, encoding='latin1')
    in_dict = in_file.tolist()
    for name in names:
        weights, biases = read_conv_params(in_dict, name)
        layers[name + '/weights'] = weights
        layers[name + '/biases'] = biases

    # transform fc6 parameters to conv6_1 parameters
    weights, biases = read_conv_params(in_dict, 'fc6')
    weights = weights.reshape((7, 7, 512, 4096))
    layers['conv6_1' + '/weights'] = weights
    layers['conv6_1' + '/biases'] = biases
    names.append('conv6_1')
    return layers, names


def inference(inputs, batch_size, is_training=True):
    '''
    Neural network for efficient piecewise crf training

    Parameters
    ----------
    inputs: tensor
        Input tensor

    batch_size: int
        Batch size

    is_training: bool
        Flag that determines whether the net is used for validation or training

    Returns
    -------
    unary: tensor
        Unary scores

    pairwise_surr_map: tensor
        Pairwise (surrounding neighbourhood) scores


    '''
    vgg_layers, vgg_layer_names = read_vgg_init(FLAGS.vgg_init_file)

    conv1_sz = 64
    conv2_sz = 128
    conv3_sz = 256
    conv4_sz = 512
    conv5_sz = 512
    conv6_1_sz = 1024
    conv6_1_kernel = 1
    conv6_sz = 512
    unary_pairwise_size = 512
    k = 3

    bn_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        'center': False,
        'scale': False,
    }

    with scopes.arg_scope([slim.ops.conv2d], stddev=0.01, weight_decay=1, is_training=is_training):
        with scopes.arg_scope([ops.conv2d], is_training=is_training):
            # FeatMap-Net

            # VGG-16 part
            net = convolve(inputs, conv1_sz, k, 'conv1_1', vgg_layers)
            net = convolve(net, conv1_sz, k, 'conv1_2', vgg_layers)
            net = ops.max_pool(net, [2, 2], scope='pool1')
            net = convolve(net, conv2_sz, k, 'conv2_1', vgg_layers)
            net = convolve(net, conv2_sz, k, 'conv2_2', vgg_layers)
            net = ops.max_pool(net, [2, 2], scope='pool2')
            net = convolve(net, conv3_sz, k, 'conv3_1', vgg_layers)
            net = convolve(net, conv3_sz, k, 'conv3_2', vgg_layers)
            net = convolve(net, conv3_sz, k, 'conv3_3', vgg_layers)
            net = ops.max_pool(net, [2, 2], scope='pool3')
            net = convolve(net, conv4_sz, k, 'conv4_1', vgg_layers)
            net = convolve(net, conv4_sz, k, 'conv4_2', vgg_layers)
            net = convolve(net, conv4_sz, k, 'conv4_3', vgg_layers)
            net = ops.max_pool(net, [2, 2], scope='pool4')
            net = convolve(net, conv5_sz, k, 'conv5_1', vgg_layers)
            net = convolve(net, conv5_sz, k, 'conv5_2', vgg_layers)
            net = convolve(net, conv5_sz, k, 'conv5_3', vgg_layers)
            net = slim.ops.max_pool(net, [2, 2], stride=1, padding='SAME', scope='pool5')

            # Additional block
            with scopes.arg_scope([ops.conv2d], batch_norm_params=bn_params):
                net = convolve(net, conv6_1_sz, conv6_1_kernel, 'conv6_1')
                net = convolve(net, conv6_sz, k, 'conv6_2')
                featmap = convolve(net, conv6_sz, k, 'conv6_3')
                unary = convolve(featmap, unary_pairwise_size, 1, 'unary_1')
                # Pairwise-Net - surrounding neighbourhood
                # reshape the featmap tensor
                pairwise_surr = tf.transpose(featmap, perm=[1, 2, 3, 0])
                pairwise_surr = tf.reshape(pairwise_surr, shape=[-1, conv6_sz, batch_size])
                first_indices = tf.gather(pairwise_surr, indices.FIRST_INDICES_SURR)
                second_indices = tf.gather(pairwise_surr, indices.SECOND_INDICES_SURR)
                pairwise_surr_map = tf.concat(1, [first_indices, second_indices])

                # try without this reshaping + transposing
                pairwise_surr_map = tf.reshape(pairwise_surr_map,
                                               shape=[indices.NUMBER_OF_NEIGHBOURS_SURR, -1, 2 * conv6_sz, batch_size])
                pairwise_surr_map = tf.transpose(pairwise_surr_map, perm=[3, 0, 1, 2])

                # apply convolution layers
                pairwise_surr_map = convolve(pairwise_surr_map, unary_pairwise_size, 1, 'pairwise_surr_1')

            # Unary-Net
            unary = convolve(unary, FLAGS.num_classes, 1, 'unary_2', activation=None)

            pairwise_surr_map = convolve(pairwise_surr_map, FLAGS.num_classes * FLAGS.num_classes, 1,
                                         'pairwise_surr_2', activation=None)

    return unary, pairwise_surr_map


def loss(out_unary, out_binary, labels_unary, labels_binary, batch_size, is_training=True):
    '''
    L2 regularized negative log likelihood

    Parameters
    ----------
    out_unary: tensor
        Unary scores

    out_binary: tensor
        Pairwise (surrounding neighbourhood) scores

    labels_unary: bool
        Unary labels

    labels_binary: tensor
        Binary labels

    batch_size: int
        Batch size

    is_training: bool
        Flag to denote whether training is being done or not

    Returns
    -------
    total_loss: float
        L2 regularized negative log likelihood


    '''
    loss_val = losses.neg_log_likelihood(out_unary, out_binary, labels_unary, labels_binary, batch_size)
    all_losses = [loss_val]
    total_loss = losses.total_loss_sum(all_losses)

    if is_training:
        loss_averages_op = losses.add_loss_summaries(total_loss)
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

    return total_loss
