# Piecewise CRF training
This folder contains scripts which implement piecewise CRF training. The implementation is based on the work of Lin et al. 

> Efficient Piecewise Training of Deep Structured Models for Semantic Segmentation <br/>
> Guosheng Lin,  Chunhua Shen, Anton van den Hengel, Ian Reid <br/>
> IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 <br/>

# Files
This folder contains various scripts used for piecewise crf learning, model evaluation, mean field inference, image preparation and generation. A short description of each script and their usage will be given shortly. Each script is very well documented, and the user is advised to look at the comments inside the scripts functions for more information about their usage.

- config
  - `prefs.py` - the main configuration file used for configuring piecewise training
  - `prefs_cityscapes_example.py` - configuration file example for the Cityscapes dataset
  - `prefs_kitti_example.py` - configuration file example for the KITTI dataset
- datasets
  - cityscapes
    - `cityscapes.py` - Cityscapes dataset meta data (labels and number of classes)
    - `prepare_dataset_files.py` - prepares all the files (images and binary files) needed for piecewise crf training as well as dense crf inference
    - `train_validation_split.py` - used for splitting the original Cityscapes train set into a train and validation subset
  - kitti
    - `kitti.py` - KITTI dataset meta data (labels and number of classes)
    - `prepare_dataset_files.py` - prepares all the files (images and binary files) needed for piecewise crf training as well as dense crf inference
    - `train_validation_split.py` - used for splitting the original KITTI train set into a train and validation subset
  - helpers
    - `pairwise_label_generator.py` - defines neighbourhoods and generates binary potentials ground truth label maps
    - `weights_generator.py` - calculates the class balancing weights for unary and binary potentials
  - `dataset.py` - contains abstract dataset definition
  - `labels.py` - contains label definitions
  - `prepare_tfrecords.py` - prepares tensorflow records used for piecewise crf training (and evaluation)
  - `reader.py` - used for reading tensorflow records and grouping them into batches
- helpers
  - `eval.py` - used for confusion matrix computation as well as IoU, pixel accuracy, recall and precision calculation
  - `io.py` - contains helper methods for loading and storing numpy arrays into binary files
  - `mean_field.py` - mean field inference implementation
  - `train.py` - contains helper methods used for printing information while training
- models
  - `losses.py` - training loss definitions
  - `piecewisecrf_model.py` - defines the neural network used for piecewise crf training
- slim
  - publicly available [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/inception/inception/slim) implementation used for simplifying the tensorflow code
- tests
  - `mean_field_test.py` - used for calculating the energy functional which mean field tries to maximize
- `eval.py` - used for evaluating the trained model on the desired dataset
- `forward_pass.py` - used for generating output images as well as unary potentials used for dense crf inference for the desired dataset
- `train.py` - used for piecewise crf training

## Training configuration
In order to configure the training parameters, the user needs to modify the `prefs.py` file. It is a python file and easily readable. It has the following structure:

```python
# Change the following parameters to suit your needs
# data preparation parameters
# used in prepare_dataset_files
flags.DEFINE_string('dataset_dir', '/home/dcausevic/datasets/cityscapes_vece_slike_new_build/',
                    'Directory containing folders created with prepare_dataset_files script')
# used in prepare_tfrecords
flags.DEFINE_string('save_dir', '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384_eval_veliki/',
                    'Directory in which tfrecord files will be saved')
# resized image dimensions
flags.DEFINE_integer('img_width', 768, 'Resized image width')
flags.DEFINE_integer('img_height', 384, 'Resized image height')
flags.DEFINE_integer('img_depth', 3, 'Resized image depth')

# training parameters
# Batch size
tf.app.flags.DEFINE_integer('batch_size', 1, '')

# Number of classes in the dataset
flags.DEFINE_integer('num_classes', 19,
                     'Number of classes in the dataset')

# Regularization factor
flags.DEFINE_float('reg_factor', 0.0005, 'Regularization factor')

# Learning rate
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 3.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('max_epochs', 15, 'Number of batches to run.')

#######################################################################################################################
#                                    Neighbourhood definitions                                                        #
#######################################################################################################################

# Surroungind neighbourhood
flags.DEFINE_integer('surrounding_neighbourhood_size', 7,
                     'Size of the surrounding neighbourhood for pairwise potentials')

# Above/below neighbourhood
flags.DEFINE_integer('neigbourhood_above_below_width', 7,
                     'Width of the above/below neighbourhood')
flags.DEFINE_integer('neigbourhood_above_below_height', 7,
                     'Height of the above/below neighbourhood')

#######################################################################################################################

# Whether to evaluate model also on the training set (SLOWS DOWN TRAINING)
tf.app.flags.DEFINE_boolean('evaluate_train_set', False, '')

# Directories used for training and validation
tf.app.flags.DEFINE_string('vgg_init_file', '/home/dcausevic/FER/caffe-tensorflow/vgg16.npy',
                           'Path to the vgg parameters file created with caffe-tensorflow')

# Results directory (model and statistics will be saved in it)
tf.app.flags.DEFINE_string('train_dir', '/home/dcausevic/Desktop/rad_results/rad_both_7x7x7_768_384_eval_veliki',
                           'Directory where to write event logs and checkpoint.')

# Records directories
tf.app.flags.DEFINE_string('train_records_dir',
                           '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384_eval_veliki/train_train/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')
tf.app.flags.DEFINE_string('val_records_dir',
                           '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384_eval_veliki/train_val/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')
tf.app.flags.DEFINE_string('test_records_dir',
                           '/home/dcausevic/datasets/cityscapes_rad_both_7x7x7_768_384_eval_veliki/val/768x384/tfrecords/',
                           'Path to the directory containing training tfrecords')

# Less likely needed to change
flags.DEFINE_float('r_mean', 123.68, 'Mean value for the red channel')
flags.DEFINE_float('g_mean', 116.779, 'Mean value for the green channel')
flags.DEFINE_float('b_mean', 103.939, 'Mean value for the blue channel')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')

# DON'T CHANGE
flags.DEFINE_integer('subsample_factor', 16, 'Subsample factor of the model')

```

In order to change a parameter, the user needs to modify the second argument in the `DEFINE_xxx` call. All parameters are well documented in the preferences file, but the more important onces are highlighted here:

- `smoothness_theta` - smoothness kernel parameter
- `smoothness_w` - smoothness kernel weight
- `appearance_theta_rgb` - appearance kernel parameter (for pixel colors)
- `appearance_theta_pos` - appearance kernel parameter (for pixel positions)
- `appearance_w` - appearance kernel weight
