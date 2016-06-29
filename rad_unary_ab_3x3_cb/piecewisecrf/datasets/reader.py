import tensorflow as tf
import piecewisecrf.datasets.cityscapes.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as label_gen

FLAGS = prefs.flags.FLAGS


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'img_name': tf.FixedLenFeature([], tf.string),
            'rgb': tf.FixedLenFeature([], tf.string),
            'class_weights': tf.FixedLenFeature([], tf.string),
            'surr_weights': tf.FixedLenFeature([], tf.string),
            'labels_unary': tf.FixedLenFeature([], tf.string),
            'labels_binary_surrounding': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['rgb'], tf.float32)
    labels_unary = tf.decode_raw(features['labels_unary'], tf.int32)
    labels_bin_sur = tf.decode_raw(features['labels_binary_surrounding'], tf.int32)
    weights = tf.decode_raw(features['class_weights'], tf.float32)
    weights_surr = tf.decode_raw(features['surr_weights'], tf.float32)
    img_name = features['img_name']

    image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_depth])
    num_pixels = FLAGS.img_height * FLAGS.img_width // FLAGS.subsample_factor // FLAGS.subsample_factor
    labels_unary = tf.reshape(labels_unary, shape=[num_pixels])
    weights = tf.reshape(weights, shape=[num_pixels])
    num_neighbours = label_gen.NUMBER_OF_NEIGHBOURS_AB
    labels_bin_sur = tf.reshape(labels_bin_sur, shape=[num_neighbours])
    weights_surr = tf.reshape(weights_surr, shape=[num_neighbours])

    return image, labels_unary, labels_bin_sur, img_name, weights, weights_surr


def inputs(dataset, shuffle=True, num_epochs=False, dataset_partition='train'):
    batch_size = FLAGS.batch_size
    if not num_epochs:
        num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(dataset.get_filenames(dataset_partition), num_epochs=num_epochs,
                                                        shuffle=shuffle,
                                                        capacity=dataset.num_examples(dataset_partition))

        image, labels_unary, labels_bin_sur, img_name, weights, weights_ab = read_and_decode(filename_queue)

        image, labels_unary, labels_bin_sur, img_name, weights, weights_ab = tf.train.batch(
            [image, labels_unary, labels_bin_sur, img_name, weights, weights_ab], batch_size=batch_size, num_threads=2,
            capacity=64)

        return image, labels_unary, labels_bin_sur, img_name, weights, weights_ab
