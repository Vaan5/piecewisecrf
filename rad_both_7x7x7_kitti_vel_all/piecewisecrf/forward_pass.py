import argparse
import os

import tensorflow as tf
import numpy as np
from tqdm import trange

import skimage
import skimage.data
import skimage.transform

import scipy.ndimage

import piecewisecrf.datasets.reader as reader
import piecewisecrf.models.piecewisecrf_model as model
import piecewisecrf.helpers.mean_field as mean_field
import piecewisecrf.datasets.kitti.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as indices
import piecewisecrf.helpers.io as io

FLAGS = prefs.flags.FLAGS


def _create_dirs(output_dir):
    '''

    Creates necessary output directory hierarchy

    Parameters:
    -----------
    output_dir: str
        Path to the output directory

    Returns
    -------
    ret_val: list
        List of paths to generated directories

    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    potentials_out_small = os.path.join(output_dir, 'unary_small')
    potentials_out_orig = os.path.join(output_dir, 'unary')
    prediction_out_small = os.path.join(output_dir, 'out_small')
    prediction_out = os.path.join(output_dir, 'out')
    prediction_ppm_out_small = os.path.join(output_dir, 'out_ppm_small')
    prediction_ppm_out = os.path.join(output_dir, 'out_ppm')

    ret_val = [potentials_out_small, potentials_out_orig, prediction_out_small,
               prediction_out, prediction_ppm_out_small, prediction_ppm_out]

    for path in ret_val:
        if not os.path.exists(path):
            os.makedirs(path)

    return ret_val


def main(argv=None):
    '''

    Creates output files for the given dataset

    '''
    possible_datasets = ['cityscapes', 'kitti']
    parser = argparse.ArgumentParser(description='Evaluates trained model on given dataset')
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Name of the dataset used for training')
    parser.add_argument('dataset_partition', type=str, choices=['train', 'validation', 'test'],
                        help='Dataset partition which will be evaluated')
    parser.add_argument('checkpoint_dir', type=str,
                        help='Path to the directory containing the trained model')
    parser.add_argument('output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    if args.dataset_name == possible_datasets[0]:
        from piecewisecrf.datasets.cityscapes.cityscapes import CityscapesDataset
        dataset = CityscapesDataset(train_dir=FLAGS.train_records_dir,
                                    val_dir=FLAGS.val_records_dir,
                                    test_dir=FLAGS.test_records_dir)
    elif args.dataset_name == possible_datasets[1]:
        from piecewisecrf.datasets.kitti.kitti import KittiDataset
        dataset = KittiDataset(train_dir=FLAGS.train_records_dir,
                               val_dir=FLAGS.val_records_dir,
                               test_dir=FLAGS.test_records_dir)

    if not os.path.exists(args.checkpoint_dir):
        print('{} was not found'.format(args.checkpoint_dir))
        exit(1)

    (potentials_out_small, potentials_out_orig, prediction_out_small,
        prediction_out, prediction_ppm_out_small, prediction_ppm_out) = _create_dirs(args.output_dir)

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if ckpt and ckpt.model_checkpoint_path:
        with tf.variable_scope('model'):
            image, labels_unary, labels_orig, labels_bin_sur, labels_bin_above_below, img_name, weights, weights_surr, weights_ab = reader.inputs(dataset,
                                                                          shuffle=False,
                                                                          dataset_partition=args.dataset_partition)
            unary_log, pairwise_log, pairwise_ab_log = model.inference(image, FLAGS.batch_size, is_training=False)

        saver = tf.train.Saver()
        print('Loading model: {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        raise ValueError()

    tf.train.start_queue_runners(sess=sess)
    print(dataset.num_examples(args.dataset_partition) // FLAGS.batch_size)
    for i in trange(dataset.num_examples(args.dataset_partition) // FLAGS.batch_size):
        logits_unary, logits_pairwise, logits_pairwise_ab, yt, names = sess.run([unary_log, pairwise_log, pairwise_ab_log, labels_unary, img_name])

        for batch in range(FLAGS.batch_size):
            print(names[batch])
            names[batch] = names[batch].decode("utf-8")
            s = mean_field.mean_field(logits_unary[batch, :, :, :],
                                      [(logits_pairwise[batch, :, :, :],
                                       list(zip(indices.FIRST_INDICES_SURR, indices.SECOND_INDICES_SURR)),
                                       indices.generate_encoding_decoding_dict(logits_unary.shape[3])[1]
                                        ),
                                      (logits_pairwise_ab[batch, :, :, :],
                                       list(zip(indices.FIRST_INDICES_AB, indices.SECOND_INDICES_AB)),
                                       indices.generate_encoding_decoding_dict(logits_unary.shape[3])[1]
                                        )
                                      ])
            s = mean_field._exp_norm(s)
            y = s.argmax(2)
            y = y.astype(np.int16)
            io.dump_nparray(y, os.path.join(prediction_out_small, "{}.bin".format(names[batch])))
            y_orig = skimage.transform.resize(y, (FLAGS.img_height, FLAGS.img_width), order=0, preserve_range=True)
            y_orig = y_orig.astype(np.int16)
            io.dump_nparray(y_orig, os.path.join(prediction_out, "{}.bin".format(names[batch])))

            yimg = np.empty((y.shape[0], y.shape[1], 3), dtype=np.uint8)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    yimg[i, j, :] = dataset.trainId2label[y[i, j]].color

            skimage.io.imsave(os.path.join(prediction_ppm_out_small, "{}.ppm".format(names[batch])), yimg)
            yimg_orig = skimage.transform.resize(yimg, (FLAGS.img_height, FLAGS.img_width),
                                                 order=0, preserve_range=True)
            yimg_orig = yimg_orig.astype(np.int16)
            skimage.io.imsave(os.path.join(prediction_ppm_out, "{}.ppm".format(names[batch])), yimg_orig)

            ret = -1.0 * np.log(s)
            ret = ret.astype(np.float32)
            io.dump_nparray(ret, os.path.join(potentials_out_small, "{}.bin".format(names[batch])))

            #ret = np.repeat(np.repeat(s, FLAGS.subsample_factor, axis=0), FLAGS.subsample_factor, axis=1)
            ret = scipy.ndimage.zoom(s, [16,16,1], order=1)
            ret = -1. * np.log(ret)
            ret = ret.astype(np.float32)
            io.dump_nparray(ret, os.path.join(potentials_out_orig, "{}.bin".format(names[batch])))


if __name__ == '__main__':
    tf.app.run()
