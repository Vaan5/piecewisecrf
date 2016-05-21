import argparse
import os

import tensorflow as tf
import numpy as np
from tqdm import trange

import skimage
import skimage.data
import skimage.transform

import piecewisecrf.models.piecewisecrf_model as model
import piecewisecrf.helpers.mean_field as mean_field
import piecewisecrf.config.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as indices

FLAGS = prefs.flags.FLAGS


def draw_prediction(y, dataset, output_dir, img_name):
    width = y.shape[1]
    height = y.shape[0]
    yimg = np.empty((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            yimg[i, j, :] = dataset.trainId2label.color

    skimage.io.imsave(os.path.join(os.path.join(output_dir, 'png'), "{}.png".format(img_name)), yimg)
    skimage.io.imsave(os.path.join(os.path.join(output_dir, 'ppm'), "{}.ppm".format(img_name)), yimg)


def save_predictions(sess, image, dataset, partition, logits_unary, logits_pairwise, output_dir):
    width = FLAGS.img_width
    height = FLAGS.img_height
    image_list = dataset.get_filenames(partition=partition)

    for i in trange(len(image_list)):
        img = skimage.data.load(image_list[i])
        img = img.astype(np.float32)

        for c in range(3):
            img[:, :, c] -= img[:, :, c].mean()
            img[:, :, c] /= img[:, :, c].std()

    img_data = img.reshape(1, height, width, 3)
    out_unary, out_pairwise = sess.run([logits_unary, logits_pairwise], feed_dict={image: img_data})

    s = mean_field.mean_field(out_unary[0, :, :, :],
                              [(out_pairwise[0, :, :, :],
                               list(zip(indices.FIRST_INDICES_SURR, indices.SECOND_INDICES_SURR)),
                               indices.generate_encoding_decoding_dict(out_unary.shape[3])[1]
                                )])
    id_img = s.argmax(2).astype(np.int32, copy=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, 'ppm')):
        os.makedirs(os.path.join(output_dir, 'ppm'))
    if not os.path.exists(os.path.join(output_dir, 'png')):
        os.makedirs(os.path.join(output_dir, 'png'))

    img_name = image_list[i][image_list[i].rfind('/') + 1:]
    img_name = img_name[0:-4]
    draw_prediction(id_img, dataset, output_dir, img_name)


def main(argv=None):
    '''
    Generates images for the given dataset
    '''
    possible_datasets = ['cityscapes', 'kitti']
    parser = argparse.ArgumentParser(description='Generates images for the given dataset')
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Name of the dataset used for training')
    parser.add_argument('dataset_partition', type=str, choices=['train', 'validation', 'test'],
                        help='Dataset partition which will be evaluated')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('model_checkpoint_path', type=str,
                        help='Path to the trained model file - model.ckpt')

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

    with tf.Graph().as_default():
        sess = tf.Session()

        batch_shape = (FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth)
        image = tf.placeholder(tf.float32, shape=batch_shape)

        # Restores from checkpoint
        with tf.Session() as sess:
            with tf.variable_scope("model"):
                unary_log, pairwise_log = model.inference(image, FLAGS.batch_size, is_training=False)

            saver = tf.train.Saver()
            saver.restore(sess, args.model_checkpoint_path)
            save_predictions(sess, image, dataset, args.dataset_partition, unary_log, pairwise_log, args.output_dir)


if __name__ == '__main__':
    tf.app.run()
