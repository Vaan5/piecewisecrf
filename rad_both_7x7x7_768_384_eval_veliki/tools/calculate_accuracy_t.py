import argparse
import os
from multiprocessing.pool import ThreadPool

import numpy as np

import piecewisecrf.helpers.io as io


def load_data(predictions_file, labels_file):
    '''

    Loads prediction and label data into numpy arrays

    Parameters
    ----------
    predictions_file: str
        Path to the prediction file

    labels_file: str
        Path to the label file

    Returns
    -------
    ret_val: tuple
        labels array, predictions array


    '''
    labels = io.load_nparray_from_bin_file(labels_file, np.uint8)
    predictions = io.load_nparray_from_bin_file(predictions_file, np.int16)

    return labels, predictions


def get_filenames(predictions_dir, labels_dir, class_ids):
    '''

    Gets ground truth and prediction file names

    Parameters
    ----------
    predictions_dir: str
        Path to the directory containing predicted images

    labels_dir: str
        Path to the directory containing ground truth files

    class_ids: list
        List of possible class ids

    Returns
    -------
    ret_val: tuple
        list of indices, list of file names, list of label file names, list of prediction file names


    '''
    label_file_names = os.listdir(labels_dir)
    file_names = sorted(label_file_names)
    label_file_names = sorted([os.path.join(labels_dir, f) for f in label_file_names])

    prediction_file_names = os.listdir(predictions_dir)
    prediction_file_names = sorted([os.path.join(predictions_dir, f) for f in prediction_file_names])

    return (range(len(label_file_names)), file_names, label_file_names,
            prediction_file_names, [class_ids] * len(label_file_names))


def evaluate_image(predicted, true):
    '''

    Calculate true positives, false positives and false negatives for one image

    Parameters
    ----------
    predicted: numpy array
        Predicted labels

    true: numpy array
        Ground truth

    Returns
    -------
    array: numpy array
        True positives, False positive, false negatives


    '''
    [c, h, w] = predicted.shape

    assert(true.shape[0] - 1 == c)
    assert(true.shape[1] == h)
    assert(true.shape[2] == w)

    keep = np.where(1 - true[-1].flatten())[0]

    predicted_flat = np.int32(np.reshape(predicted, [c, h * w])[:, keep])
    true_flat = np.int32(np.reshape(true[:-1], [c, h * w])[:, keep])

    tp = np.sum(predicted_flat * true_flat, axis=1)[None]                   # [1,c]
    fp = np.sum(predicted_flat * (1 - true_flat), axis=1)[None]             # [1,c]
    fn = np.sum((1 - predicted_flat) * true_flat, axis=1)[None]             # [1,c]

    return np.vstack([tp, fp, fn])                                          # [3,c]


def run(zipped):
    '''

    Runs evaluation for one image

    '''
    i, file, l, p, class_ids = zipped
    labels, predictions = load_data(p, l)

    # no slice for negative class
    predicted = np.vstack([(predictions == class_id)[None] for class_id in class_ids])
    true = np.vstack([(labels == class_id)[None] for class_id in class_ids])
    true = np.vstack([true, (labels > class_ids[-1])[None]])   # ignore don't care classes

    return evaluate_image(predicted, true)[None]


def evaluate_segmentation(predictions_dir, labels_dir, dataset):
    '''

    Runs evaluation for all images

    Parameters
    ----------
    predictions_dir: str
        Directory containing prediction files

    labels_dir: str
        Directory containing label files

    dataset: Dataset
        Dataset for which evaluation is being done

    Returns
    -------
    ret_val: list
        Per class iou, per pixel accuracy, segmentation statistics,
        pixel accuracy, per class accuracy, mean iou


    '''
    class_ids = dataset.classes
    zipped = zip(*get_filenames(predictions_dir, labels_dir, class_ids))

    segmentation_stats = ThreadPool().map(run, zipped)

    segmentation_stats = np.concatenate(segmentation_stats, axis=0)
    aggregated_stats_class = np.float32(np.sum(segmentation_stats, axis=0))                     # [3,c]

    tpfn = np.sum(np.concatenate([aggregated_stats_class[0][None],
                                  aggregated_stats_class[2][None]], axis=0), axis=0)
    tpfp = np.sum(np.concatenate([aggregated_stats_class[0][None],
                                  aggregated_stats_class[1][None]], axis=0), axis=0)

    recall_per_class = aggregated_stats_class[0] / tpfn                                  # [c]
    precision_per_class = aggregated_stats_class[0] / tpfp                                  # [c]
    per_class_iou = aggregated_stats_class[0] / np.sum(aggregated_stats_class, axis=0)     # [c]
    per_class_iou = np.nan_to_num(per_class_iou)

    aggregated_stats_all = np.sum(aggregated_stats_class, axis=1)                               # [3]
    per_pixel_accuracy = aggregated_stats_all[0] / (aggregated_stats_all[0] + aggregated_stats_all[1])

    real_per_pixel_accuracy = aggregated_stats_all[0] / (aggregated_stats_all[0] + aggregated_stats_all[2])
    iou = 100.0 * per_class_iou.mean()

    return [per_class_iou, per_pixel_accuracy, segmentation_stats,
            precision_per_class, recall_per_class, iou]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates semantic segmentation")
    parser.add_argument('predictions_dir', type=str,
                        help="Directory containing predicted files")
    parser.add_argument('labels_dir', type=str,
                        help="Directory containing ground truth files")
    possible_datasets = ['cityscapes', 'kitti']
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Name of the dataset used for evaluation')

    args = parser.parse_args()

    if args.dataset_name == possible_datasets[0]:
        from piecewisecrf.datasets.cityscapes.cityscapes import CityscapesDataset
        dataset = CityscapesDataset()
    elif args.dataset_name == possible_datasets[1]:
        from piecewisecrf.datasets.kitti.kitti import KittiDataset
        dataset = KittiDataset()

    results = evaluate_segmentation(args.predictions_dir, args.labels_dir, dataset)

    print('IoU = {:.2f}'.format(results[5]))
    print('Pixel Accuracy = {:.2f}'.format(100 * results[1]))
    print('Recall = {:.2f}'.format(100 * results[4].mean()))
    print('Precision = {}'.format(100 * results[3].mean()))
    print('Per class IOU = {}'.format(100.0 * results[0]))
