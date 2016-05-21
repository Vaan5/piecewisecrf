import os
import time
import argparse
from tqdm import trange

import numpy as np
import tensorflow as tf

import piecewisecrf.helpers.train as train_helper
import piecewisecrf.helpers.eval as eval_helper
import piecewisecrf.datasets.reader as reader
import piecewisecrf.helpers.mean_field as mean_field
import piecewisecrf.models.piecewisecrf_model as model
import piecewisecrf.datasets.cityscapes.prefs as prefs
import piecewisecrf.datasets.helpers.pairwise_label_generator as indices

FLAGS = prefs.flags.FLAGS


def evaluate(sess, unary, pairwise, loss, labels_unary, dataset, dataset_partition='validation'):
    '''

    Evaluate trained model on the validation dataset

    Parameters
    ----------
    session : tf.Session object
        Current tensorflow session

    unary: tensorflow operation
        Operation used to calculate the unary potentials output of the net

    pairwise: tensorflow operation
        Operation used to calculate the pairwise potentials output of the net

    loss: tensorflow operation
        Operation used to calculate the loss of the net

    labels_unary: tensorflow operation
        Operation used to get the labels for the given dataset

    dataset: Dataset object
        Dataset used for validating the trained model

    dataset_partition: str
        Subset of the original dataset (train, validation, test)

    Returns
    -------
    loss_avg: float
        Average net loss on the given dataset

    pixel_acc: float
        Average pixel accuracy on the given dataset

    iou_acc: float
        Average IOU on the given dataset

    recall: float
        Average recall on the given dataset

    precision: float
        Average precision on the given dataset


    '''
    conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    loss_avg = 0.0

    for step in trange(int((dataset.num_examples(dataset_partition) / FLAGS.batch_size))):
        scores_unary, scores_pairwise, loss_val, yt_unary = sess.run([unary, pairwise, loss, labels_unary])
        for batch in range(FLAGS.batch_size):
            s = mean_field.mean_field(scores_unary[batch, :, :, :],
                                      [(scores_pairwise[batch, :, :, :],
                                       list(zip(indices.FIRST_INDICES_AB, indices.SECOND_INDICES_AB)),
                                       indices.generate_encoding_decoding_dict(scores_unary.shape[3])[1]
                                        )])
            label_map = s.argmax(2).astype(np.int32, copy=False)
            eval_helper.confusion_matrix(label_map.reshape(-1),
                                         yt_unary[batch], conf_mat,
                                         dataset.num_classes())
            loss_avg += loss_val
    print("Total score")

    pixel_acc, iou_acc, recall, precision, _ = eval_helper.compute_errors(conf_mat,
                                                                          dataset_partition.capitalize(),
                                                                          dataset.trainId2label,
                                                                          verbose=True)

    return loss_avg / dataset.num_examples(dataset_partition), pixel_acc, iou_acc, recall, precision


def train(dataset, resume_path=None):
    '''

    Trains the network on the given dataset

    Parameters
    ----------
    dataset : Dataset object
        Dataset for which training is being done

    resume_path: str
        Path to partially trained model


    '''
    with tf.Graph().as_default():

        # configure the training session
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=config)

        # number of train() calls
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (dataset.num_examples('train') /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Get images and labels for training and validation
        image, labels_unary, labels_bin_sur, img_name = reader.inputs(dataset,
                                                                      shuffle=True,
                                                                      num_epochs=FLAGS.max_epochs,
                                                                      dataset_partition='train')
        image_val, labels_unary_val, labels_bin_sur_val, img_name_val = reader.inputs(dataset,
                                                                                      shuffle=False,
                                                                                      num_epochs=FLAGS.max_epochs,
                                                                                      dataset_partition='validation')
        image_train, labels_unary_train, labels_bin_sur_train, img_name_train = reader.inputs(
                                                                                    dataset,
                                                                                    shuffle=False,
                                                                                    num_epochs=FLAGS.max_epochs,
                                                                                    dataset_partition='train')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        with tf.variable_scope('model'):
            unary_log, pairwise_log = model.inference(image, FLAGS.batch_size)
            # Calculate loss.
            loss = model.loss(unary_log, pairwise_log, labels_unary, labels_bin_sur, FLAGS.batch_size)

        with tf.variable_scope('model', reuse=True):
            unary_log_val, pairwise_log_val = model.inference(image_val, FLAGS.batch_size, is_training=False)
            # Calculate loss.
            loss_val = model.loss(unary_log_val, pairwise_log_val, labels_unary_val,
                                  labels_bin_sur_val, FLAGS.batch_size, is_training=False)

        # used for validating performance on train set
        with tf.variable_scope('model', reuse=True):
            unary_log_train, pairwise_log_train = model.inference(image_train, FLAGS.batch_size, is_training=False)
            # Calculate loss.
            loss_train = model.loss(unary_log_train, pairwise_log_train, labels_unary_train,
                                  labels_bin_sur_train, FLAGS.batch_size, is_training=False)

        # Add a summary to track the learning rate.
        tf.scalar_summary('learning_rate', lr)

        # Compute gradients.
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.histogram_summary(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.max_epochs)

        if resume_path is None:
            # Build an initialization operation to run
            init = tf.initialize_all_variables()
            sess.run(init)
        else:
            assert tf.gfile.Exists(resume_path)
            saver.restore(sess, resume_path)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)

        variable_map = train_helper.get_variable_map()
        loss_avg_train = variable_map['model/model/total_loss/avg:0']

        train_iou_data = []
        train_pixacc_data = []

        val_iou_data = []
        val_pixacc_data = []
        train_loss_data = []
        val_loss_data = []

        ex_start_time = time.time()
        for epoch_num in range(1, FLAGS.max_epochs + 1):
            conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)

            for step in range(int(num_batches_per_epoch)):
                start_time = time.time()
                if step % 100 == 0:
                    ret_val = sess.run([train_op, loss, loss_avg_train, unary_log,
                                        pairwise_log, labels_unary, labels_bin_sur, lr, img_name,
                                        global_step, summary_op])
                    # unpack returned values
                    (_, loss_value, train_loss_avg, scores_unary,
                        scores_pairwise_surr, yt_unary, yt_pairwise_surr,
                        clr, filename, global_step_value, summary_str) = ret_val

                    summary_writer.add_summary(summary_str, global_step_value)
                else:
                    ret_val = sess.run([train_op, loss, unary_log,
                                        pairwise_log, labels_unary,
                                        labels_bin_sur, lr, img_name,
                                        global_step])
                    # unpack returned values
                    (_, loss_value, scores_unary,
                        scores_pairwise_surr, yt_unary, yt_pairwise_surr,
                        clr, filename, global_step_value) = ret_val

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step == 0:
                    print('lr = {}'.format(clr))
                if step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = '%s: epoch %d, step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (train_helper.get_expired_time(ex_start_time), epoch_num, step,
                                        loss_value, examples_per_sec, sec_per_batch))

            train_iou = 0.0
            train_pixacc = 0.0
            train_loss = 0.0
            if FLAGS.evaluate_train_set:
                train_loss, train_pixacc, train_iou, train_recall, train_precision = evaluate(
                                                                            sess, unary_log_train,
                                                                            pairwise_log_train, loss_train,
                                                                            labels_unary_train, dataset,
                                                                            dataset_partition='train'
                                                                            )

            # evaluate model on the validation set
            val_loss, val_pixacc, val_iou, recall, precision = evaluate(sess, unary_log_val,
                                                                        pairwise_log_val, loss_val,
                                                                        labels_unary_val, dataset)
            val_iou_data += [val_iou]
            val_pixacc_data += [val_pixacc]
            val_loss_data += [val_loss]
            train_iou_data += [train_iou]
            train_pixacc_data += [train_pixacc]
            train_loss_data += [train_loss]

            if epoch_num > 1:
                print('Best IoU = ', max(val_iou_data))

                eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
                                                   [train_loss_data, val_loss_data],
                                                   [train_iou_data, val_iou_data],
                                                   [train_pixacc_data, val_pixacc_data])

            # Save the model checkpoint periodically.
            if val_iou >= max(val_iou_data):
                print('Saving model...')
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    '''

    Trains neural network for efficient piecewise crf training

    '''
    print('Results dir: {}'.format(FLAGS.train_dir))
    print('Train dataset dir: {}'.format(FLAGS.train_records_dir))
    print('Validation dataset dir: {}'.format(FLAGS.val_records_dir))

    possible_datasets = ['cityscapes', 'kitti']
    parser = argparse.ArgumentParser(description='Trains neural network')
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Name of the dataset used for training')
    parser.add_argument('--resume', dest='resume', type=str, help='Path to the partially trained model')

    args = parser.parse_args()

    if args.dataset_name == possible_datasets[0]:
        from piecewisecrf.datasets.cityscapes.cityscapes import CityscapesDataset
        dataset = CityscapesDataset(train_dir=FLAGS.train_records_dir,
                                    val_dir=FLAGS.val_records_dir)
    elif args.dataset_name == possible_datasets[1]:
        from piecewisecrf.datasets.kitti.kitti import KittiDataset
        dataset = KittiDataset(train_dir=FLAGS.train_records_dir,
                               val_dir=FLAGS.val_records_dir)

    resume_path = None
    if args.resume and os.path.exists(args.resume):
        resume_path = args.resume

    train(dataset, resume_path)


if __name__ == '__main__':
    tf.app.run()
