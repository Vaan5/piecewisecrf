import tensorflow as tf


flags = tf.app.flags

# data preparation parameters
flags.DEFINE_string('dataset_dir', '/home/dcausevic/datasets/cityscapes_vece_slike/',
                    'Directory containing folders created with prepare_dataset_files script')
flags.DEFINE_string('save_dir', '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384/',
                    'Directory in which tfrecord files will be saved')
flags.DEFINE_integer('img_width', 768, 'Resized image width')
flags.DEFINE_integer('img_height', 384, 'Resized image height')
flags.DEFINE_integer('img_depth', 3, 'Resized image depth')
flags.DEFINE_float('r_mean', 123.68, 'Mean value for the red channel')
flags.DEFINE_float('g_mean', 116.779, 'Mean value for the green channel')
flags.DEFINE_float('b_mean', 103.939, 'Mean value for the blue channel')

# training parameters

tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_string('vgg_init_file', '/home/dcausevic/FER/caffe-tensorflow/vgg16.npy',
                           'Path to the vgg parameters file created with caffe-tensorflow')
flags.DEFINE_integer('num_classes', 19,
                     'Number of classes in the dataset')
flags.DEFINE_float('reg_factor', 0.0005, 'Regularization factor')

tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')

tf.app.flags.DEFINE_string('train_dir', '/home/dcausevic/Desktop/rad_results/rad_both_7x7x7_768_384',
                           'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer('max_epochs', 20, 'Number of batches to run.')
tf.app.flags.DEFINE_string('train_records_dir', '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384/train_train/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')
tf.app.flags.DEFINE_string('val_records_dir', '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384/train_val/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')
tf.app.flags.DEFINE_string('test_records_dir', '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384/val/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')
tf.app.flags.DEFINE_boolean('is_training', True, '')
tf.app.flags.DEFINE_boolean('evaluate_train_set', False, '')

# other parameters
flags.DEFINE_integer('subsample_factor', 16, 'Subsample factor of the model')

# evaluation parameters
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/dcausevic/Desktop/rad_results/rad_both_7x7x7_768_384', 'Path to the directory containing trained models')


#######################################################################################################################
####                                    Neighbourhood definitions                                                  ####
#######################################################################################################################

# Surroungind neighbourhood
flags.DEFINE_integer('surrounding_neighbourhood_size', 7,
                     'Size of the surrounding neighbourhood for pairwise potentials')

# Above/below neighbourhood
flags.DEFINE_integer('neigbourhood_above_below_width', 7,
                     'Width of the above/below neighbourhood')
flags.DEFINE_integer('neigbourhood_above_below_height', 7,
                     'Height of the above/below neighbourhood')
