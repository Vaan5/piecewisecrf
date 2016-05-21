import argparse
import os
import random
import shutil
import tqdm

import numpy as np

import skimage
import skimage.data

from piecewisecrf.datasets.kitti.kitti import KittiDataset


def _pick_images(labels_dir, size):
    '''

    Randomly selects size images from labels_dir

    Parameters
    ----------
    labels_dir: str
        Path to the directory containing label images

    size: int
        Number of images to pick

    Returns
    -------
    array: numpy array
        Picked image names


    '''
    images = os.listdir(labels_dir)
    return random.sample(images, size)


def _label_statistics(image_paths):
    '''

    Calculates label statistics (number of picked pixels for each class)

    Parameters
    ----------
    image_paths : list
        List of absolute paths for picked images

    Returns
    -------
    array: numpy array
        Number of selected pixels per class


    '''
    ds = KittiDataset()

    def _rgb_2_label(rgb):
        return ds.color2label[tuple(rgb)].trainId

    total_counts = np.zeros(ds.num_classes())
    for img in image_paths:
        rgb = skimage.data.load(img)
        labels = np.apply_along_axis(_rgb_2_label, 2, rgb)
        indices, counts = np.unique(labels, return_counts=True)
        if indices[-1] >= ds.num_classes():
            indices = indices[0:-1]
            counts = counts[0:-1]
        total_counts[indices] += counts
    return total_counts


def main(dataset_dir, size):
    '''

    Splits the train dataset into train and validation subsets

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset folder (containing train and val subsets)

    size: int
        Wanted size of the validation subset


    '''
    label_input_dir = os.path.join(dataset_dir, 'train')
    label_input_dir = os.path.join(label_input_dir, 'labels')

    img_input_dir = os.path.join(dataset_dir, 'train')
    img_input_dir = os.path.join(img_input_dir, 'data')
    img_input_dir = os.path.join(img_input_dir, 'rgb')

    picked_all_labels = False
    while not picked_all_labels:
        picked_images = _pick_images(label_input_dir, size)
        stats = _label_statistics([os.path.join(label_input_dir, x) for x in picked_images])
        print("Pixels per class: {}".format(stats))
        if np.count_nonzero(stats) == KittiDataset().num_classes():
            picked_all_labels = True

    print("Processing images")
    print("Creating train_train and train_val folders")
    train_path = os.path.join(dataset_dir, 'train_train')
    val_path = os.path.join(dataset_dir, 'train_val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    train_data_path = os.path.join(train_path, 'data')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    train_data_path = os.path.join(train_data_path, 'data')
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    train_label_path = os.path.join(train_path, 'labels')
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    val_data_path = os.path.join(val_path, 'data')
    if not os.path.exists(val_data_path):
        os.makedirs(val_data_path)
    val_data_path = os.path.join(val_data_path, 'data')
    if not os.path.exists(val_data_path):
        os.makedirs(val_data_path)
    val_label_path = os.path.join(val_path, 'labels')
    if not os.path.exists(val_label_path):
        os.makedirs(val_label_path)
    print("Finished creating folders")

    print("Moving image and label files")
    picked_images = set(picked_images)
    for img in tqdm.tqdm(os.listdir(img_input_dir)):
        if img in picked_images:
            shutil.copy(os.path.join(img_input_dir, img), os.path.join(val_data_path, img))
            shutil.copy(os.path.join(label_input_dir, img), os.path.join(val_label_path, img))
        else:
            shutil.copy(os.path.join(img_input_dir, img), os.path.join(train_data_path, img))
            shutil.copy(os.path.join(label_input_dir, img), os.path.join(train_label_path, img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits train set into train and validation sets.')
    parser.add_argument('dataset_path', type=str, help='Path to the root directory of the kitti dataset'
                        ' (contains train and valid subdirectories)')
    parser.add_argument('size', type=int, help='Number of images in the validation set')

    args = parser.parse_args()
    main(args.dataset_path, args.size)
