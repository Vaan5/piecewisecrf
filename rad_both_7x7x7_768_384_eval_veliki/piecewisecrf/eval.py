import argparse
import os

import tensorflow as tf
import numpy as np
import scipy.ndimage
from tqdm import trange

import piecewisecrf.datasets.reader as reader
import piecewisecrf.models.piecewisecrf_model as model
import piecewisecrf.helpers.mean_field as mean_field
import piecewisecrf.datasets.cityscapes.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as indices
import piecewisecrf.helpers.eval as eval_helper

FLAGS = prefs.flags.FLAGS


def main(argv=None):
    '''
    Evaluates the trained model on the specified dataset
    '''
    possible_datasets = ['cityscapes', 'kitti']
    parser = argparse.ArgumentParser(description='Evaluates trained model on given dataset')
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Name of the dataset used for training')
    parser.add_argument('dataset_partition', type=str, choices=['train', 'validation', 'test'],
                        help='Dataset partition which will be evaluated')
    parser.add_argument('checkpoint_dir', type=str,
                        help='Path to the directory containing the trained model')
    parser.add_argument('--calculate_energy', help='Detailed output',
                        dest='calculate_energy', action='store_true')

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

    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    if ckpt and ckpt.model_checkpoint_path:
        with tf.variable_scope('model'):
            image, labels_unary, lo, labels_bin_sur, labels_bin_above_below, img_name, weights, weights_surr, weights_ab = reader.inputs(dataset,
                                                                          shuffle=False,
                                                                          dataset_partition=args.dataset_partition)
            unary_log, pairwise_log, pairwise_ab_log = model.inference(image, FLAGS.batch_size, is_training=False)

        saver = tf.train.Saver()
        print('Loading model: {}'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        raise ValueError()

    conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    conf_mat_without_mf = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    tf.train.start_queue_runners(sess=sess)
    for i in trange(dataset.num_examples(args.dataset_partition) // FLAGS.batch_size):
        logits_unary, logits_pairwise, logits_pairwise_ab, yt, names = sess.run([unary_log, pairwise_log, pairwise_ab_log, lo, img_name])

        for batch in range(FLAGS.batch_size):
            s = mean_field.mean_field(logits_unary[batch, :, :, :],
                                      [(logits_pairwise[batch, :, :, :],
                                       list(zip(indices.FIRST_INDICES_SURR, indices.SECOND_INDICES_SURR)),
                                       indices.generate_encoding_decoding_dict(logits_unary.shape[3])[1]
                                        ),(logits_pairwise_ab[batch, :, :, :],
                                       list(zip(indices.FIRST_INDICES_AB, indices.SECOND_INDICES_AB)),
                                       indices.generate_encoding_decoding_dict(logits_unary.shape[3])[1]
                                        )],
                                      calculate_energy=args.calculate_energy)
            s = scipy.ndimage.zoom(s, [16,16,1], order=1)
            logits_unary = scipy.ndimage.zoom(logits_unary,[1,16,16,1], order=1)
            yk = logits_unary[batch].argmax(2)
            y = s.argmax(2)
            eval_helper.confusion_matrix(y.reshape(-1), yt[batch], conf_mat, dataset.num_classes())
            eval_helper.confusion_matrix(yk.reshape(-1), yt[batch], conf_mat_without_mf, dataset.num_classes())

            print(names[batch])

            eval_helper.compute_errors(conf_mat, args.dataset_partition, dataset.trainId2label)
            eval_helper.compute_errors(conf_mat_without_mf, '{} without mf'.format(args.dataset_partition),
                                       dataset.trainId2label)
    eval_helper.compute_errors(conf_mat, args.dataset_partition, dataset.trainId2label)


if __name__ == '__main__':
    tf.app.run()
