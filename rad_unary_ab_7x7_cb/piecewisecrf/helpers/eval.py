import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def confusion_matrix(y, yt, conf_mat, max_label):
    '''

    Calculates confusion matrix

    Parameters
    ----------
    y : numpy array
        Predictions

    yt : numpy array
        Ground truth

    conf_mat : numpy array
        Confusion matrix to be filled

    max_label: int
        Maximum label index to be considered


    '''
    for i in range(y.size):
        l = y[i]
        lt = yt[i]
        if lt >= 0 and lt < max_label:
            conf_mat[l, lt] += 1


def compute_errors(conf_mat, name, class_info, verbose=True):
    '''

    Computes IoU, pixel accuracy, precision, recall

    Parameters
    ----------
    conf_mat : numpy array
        Confusion matrix

    name : str
        Dataset partition name for which the calculation is being done

    class_info : dict
        TrainId to Label mapping

    verbose: bool
        Flag used for determining output

    Returns
    -------
    avg_pixel_acc: float
        Average pixel accuracy on the given dataset

    avg_class_iou: float
        Average IoU on the given dataset

    avg_class_recall: float
        Average recall on the given dataset

    avg_class_precision: float
        Average precision on the given dataset

    total_size: int
        Number of considered pixels


    '''
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()

    # pixel accuracy
    avg_pixel_acc = 100.0 * num_correct / total_size

    # tp, fp, fn + combinations
    tpfn = conf_mat.sum(0)
    tpfp = conf_mat.sum(1)
    fn = tpfn - conf_mat.diagonal()
    fp = tpfp - conf_mat.diagonal()

    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    if verbose:
        print(name + ' errors:')

    for i in range(num_classes):
        tp = conf_mat[i, i]
        if (tp + fp[i] + fn[i]) > 0:
            class_iou[i] = 100.0 * tp / (tp + fp[i] + fn[i])
        if tpfn[i] > 0:
            class_recall[i] = 100.0 * tp / tpfn[i]
        if tpfp[i] > 0:
            class_precision[i] = 100.0 * tp / tpfp[i]

        class_name = class_info[i].name
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))

    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()

    if verbose:
        print(name + ' pixel accuracy = %.2f %%' % avg_pixel_acc)
        print(name + ' IoU mean class accuracy - TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print(name + ' mean class recall - TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print(name + ' mean class precision - TP / (TP+FP) = %.2f %%' % avg_class_precision)

    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size


def plot_training_progress(save_dir, loss, iou, pixel_acc):
    '''

    Plots loss, IoU, pixel accuracy

    Parameters
    ----------
    save_dir : str
        Directory in which the pdf with the plot will be saved

    loss : list
        List of loss values for plotting

    iou : list
        List of IoUs for plotting

    pixel_acc: list
        List of pixel accuracies for plotting


    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 6
    title_size = 10
    train_color = 'm'
    val_color = 'c'

    x_data = np.linspace(1, len(loss[0]), len(loss[0]))
    ax1.set_title('loss', fontsize=title_size)
    ax1.plot(x_data, loss[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
             label='train')
    ax1.plot(x_data, loss[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
             label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)
    ax2.set_title('IoU')
    ax2.plot(x_data, iou[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
             label='train')
    ax2.plot(x_data, iou[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
             label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)
    ax3.set_title('pixel accuracy')
    ax3.plot(x_data, pixel_acc[0], marker='o', color=train_color, linewidth=linewidth, linestyle='-',
             label='train')
    ax3.plot(x_data, pixel_acc[1], marker='o', color=val_color, linewidth=linewidth, linestyle='-',
             label='validation')
    ax3.legend(loc='upper left', fontsize=legend_size)

    save_path = os.path.join(save_dir, 'training_plot.pdf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Plotting in: ', save_path)
    plt.savefig(save_path)
    plt.close(fig)
