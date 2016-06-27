import os
import tqdm

import skimage
import skimage.data
import skimage.transform

import numpy as np

import tensorflow as tf

import piecewisecrf.helpers.io as io
import piecewisecrf.config.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as label_gen
import piecewisecrf.datasets.helpers.weights_generator as weights_gen

FLAGS = prefs.flags.FLAGS


def _int64_feature(value):
    """

    Wrapper for inserting int64 features into Example proto.

    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """

    Wrapper for inserting bytes features into Example proto.

    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def prepare_dataset(name):
    '''

    Generates tfrecords for model training and validation.
    Uses outputs generated with prepare_dataset_files.py

    Parameters
    ----------
    name : str
        Name of a subset from the original dataset
        (corresponds to subdirectories generated with prepare_dataset_files.py)


    Used FLAGS
    ------------
        dataset_dir
        save_dir

        img_width
        img_height

        r_mean
        g_mean
        b_mean

        subsample_factor
        num_classes


    '''
    print('Prepairing {} dataset'.format(name))

    # generate paths to input directories
    root_dir = os.path.join(FLAGS.dataset_dir, name)
    root_dir = os.path.join(root_dir, '{}x{}'.format(FLAGS.img_width, FLAGS.img_height))
    img_dir = os.path.join(root_dir, 'img')
    gt_dir = os.path.join(root_dir, 'gt_bin')

    # output directory path
    save_dir = os.path.join(FLAGS.save_dir, name)
    save_dir = os.path.join(save_dir, '{}x{}'.format(FLAGS.img_width, FLAGS.img_height))
    save_dir = os.path.join(save_dir, 'tfrecords')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rgb_means = [FLAGS.r_mean, FLAGS.g_mean, FLAGS.b_mean]

    for img_name in tqdm.tqdm(next(os.walk(img_dir))[2]):
        img_prefix = img_name[0:img_name.index('.')]
        rgb = skimage.data.load(os.path.join(img_dir, img_name))
        rgb = rgb.astype(np.float32)

        # needed cause we use VGG16
        for c in range(3):
            rgb[:, :, c] -= rgb_means[c]

        labels = io.load_nparray_from_bin_file(os.path.join(gt_dir, '{}.bin'.format(img_prefix)), np.uint8)
        labels_orig = labels.astype(np.int32)
        subslampled_size = (labels.shape[0] / FLAGS.subsample_factor, labels.shape[1] / FLAGS.subsample_factor)
        labels = skimage.transform.resize(labels, subslampled_size, order=0, preserve_range=True)
        labels = labels.astype(np.int32)  # likely not needed
        label_weights, class_hist = weights_gen.calculate_weights(labels, FLAGS.num_classes)
        labels_pairwise_surrounding = label_gen.generate_pairwise_labels(labels,
                                                                         label_gen.get_indices_surrounding,
                                                                         FLAGS.num_classes)
        labels_surr_weights = weights_gen.calculate_weights_binary(class_hist,
                                                                   labels_pairwise_surrounding,
                                                                   label_gen.generate_encoding_decoding_dict(
                                                                       FLAGS.num_classes)[1],
                                                                   FLAGS.num_classes
                                                                   )
        labels_pairwise_above_below = label_gen.generate_pairwise_labels(labels,
                                                                         label_gen.get_indices_above_below,
                                                                         FLAGS.num_classes)
        labels_ab_weights = weights_gen.calculate_weights_binary(class_hist,
                                                                 labels_pairwise_above_below,
                                                                 label_gen.generate_encoding_decoding_dict(
                                                                     FLAGS.num_classes)[1],
                                                                 FLAGS.num_classes
                                                                 )

        rows = rgb.shape[0]
        cols = rgb.shape[1]
        depth = rgb.shape[2]

        filename = os.path.join(save_dir, '{}.tfrecords'.format(img_prefix))
        writer = tf.python_io.TFRecordWriter(filename)

        rgb_raw = rgb.tostring()
        labels_raw = labels.tostring()
        labels_orig_raw = labels_orig.tostring()
        labels_pairwise_surrounding_raw = labels_pairwise_surrounding.tostring()
        labels_pairwise_above_below_raw = labels_pairwise_above_below.tostring()
        weights_str = label_weights.tostring()
        weights_surr_str = labels_surr_weights.tostring()
        weights_ab_str = labels_ab_weights.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'img_name': _bytes_feature(img_prefix.encode()),
            'rgb': _bytes_feature(rgb_raw),
            'class_weights': _bytes_feature(weights_str),
            'surr_weights': _bytes_feature(weights_surr_str),
            'ab_weights': _bytes_feature(weights_ab_str),
            'labels_unary': _bytes_feature(labels_raw),
            'labels_orig': _bytes_feature(labels_orig_raw),
            'labels_binary_surrounding': _bytes_feature(labels_pairwise_surrounding_raw),
            'labels_binary_above_below': _bytes_feature(labels_pairwise_above_below_raw)}))
        writer.write(example.SerializeToString())
        writer.close()

        if name == 'train_train':
            filename_flip = os.path.join(save_dir, '{}_flip.tfrecords'.format(img_prefix))
            writer = tf.python_io.TFRecordWriter(filename_flip)

            rgb = np.fliplr(rgb)
            rgb_raw = rgb.tostring()
            labels = np.fliplr(labels)
            labels_orig = np.fliplr(labels_orig)
            labels_orig_raw = labels_orig.tostring()
            labels_raw = labels.tostring()
            labels_pairwise_surrounding = label_gen.generate_pairwise_labels(labels,
                                                                             label_gen.get_indices_surrounding,
                                                                             FLAGS.num_classes)
            label_weights, class_hist = weights_gen.calculate_weights(labels, FLAGS.num_classes)
            labels_surr_weights = weights_gen.calculate_weights_binary(class_hist,
                                                                       labels_pairwise_surrounding,
                                                                       label_gen.generate_encoding_decoding_dict(
                                                                           FLAGS.num_classes)[1],
                                                                       FLAGS.num_classes
                                                                       )
            labels_pairwise_above_below = label_gen.generate_pairwise_labels(labels,
                                                                             label_gen.get_indices_above_below,
                                                                             FLAGS.num_classes)
            labels_ab_weights = weights_gen.calculate_weights_binary(class_hist,
                                                                     labels_pairwise_above_below,
                                                                     label_gen.generate_encoding_decoding_dict(
                                                                         FLAGS.num_classes)[1],
                                                                     FLAGS.num_classes
                                                                     )
            labels_pairwise_surrounding_raw = labels_pairwise_surrounding.tostring()
            labels_pairwise_above_below_raw = labels_pairwise_above_below.tostring()
            weights_str = label_weights.tostring()
            weights_surr_str = labels_surr_weights.tostring()
            weights_ab_str = labels_ab_weights.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'img_name': _bytes_feature(img_prefix.encode()),
                'rgb': _bytes_feature(rgb_raw),
                'class_weights': _bytes_feature(weights_str),
                'surr_weights': _bytes_feature(weights_surr_str),
                'ab_weights': _bytes_feature(weights_ab_str),
                'labels_unary': _bytes_feature(labels_raw),
                'labels_orig': _bytes_feature(labels_orig_raw),
                'labels_binary_surrounding': _bytes_feature(labels_pairwise_surrounding_raw),
                'labels_binary_above_below': _bytes_feature(labels_pairwise_above_below_raw)}))
            writer.write(example.SerializeToString())
            writer.close()


def main(argv):
    '''

    Generates tfrecords the train_val, train_train and val subsets.

    '''
    prepare_dataset('train_val')
    prepare_dataset('train_train')
    prepare_dataset('val')


if __name__ == '__main__':
    tf.app.run()
