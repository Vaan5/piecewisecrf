import argparse
import os
import tqdm

import skimage
import skimage.data
import skimage.transform

import numpy as np

import piecewisecrf.helpers.io as io
from piecewisecrf.datasets.kitti.kitti import KittiDataset


def _create_folder(root_dir, sub_dir):
    '''

    Creates a subfolder sub_dir in root_dir

    Parameters
    ----------
    root_dir : str
        Path to the root directory

    sub_dir: str
        Subdirectory name


    '''
    dir_path = os.path.join(root_dir, sub_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def _create_folders(root_dir, sub_dir):
    '''

    Creates the following folder hierarchy inside root_dir

    root_dir
        | sub_dir
            | gt
            | img
            | imgppm
            | gt_ppm
            | gt_bin

    Parameters
    ----------
    root_dir : str
        Path to the root directory

    sub_dir: str
        Subdirectory name


    '''
    output_dir = _create_folder(root_dir, sub_dir)

    label_dest_dir = _create_folder(output_dir, 'gt')
    image_dest_dir = _create_folder(output_dir, 'img')
    image_dest_ppm_dir = _create_folder(output_dir, 'img_ppm')
    label_dest_ppm_dir = _create_folder(output_dir, 'gt_ppm')
    label_dest_bin_dir = _create_folder(output_dir, 'gt_bin')

    return label_dest_dir, image_dest_dir, image_dest_ppm_dir, label_dest_ppm_dir, label_dest_bin_dir


def main(dataset_dir, output_dir, subset, resize, onlyresize):
    '''

    Prepares all the necessary data for the semantic segmentation pipeline

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory

    output_dir: str
        Path to the output directory directory

    subset: str
        Dataset subset (train, train_val, train_train, valid)

    resize: list
        [width, height] for the resized images

    onlyresize: bool
        Whether only resized images will be created


    '''
    print("Creating directories")
    subset_dir = os.path.join(dataset_dir, subset)
    image_input_dir = os.path.join(subset_dir, 'data')
    image_input_dir = os.path.join(image_input_dir, 'rgb')
    label_input_dir = os.path.join(subset_dir, 'labels')
    output_dir = _create_folder(output_dir, subset)

    # create folder hierarchy for original size images
    if not onlyresize:
        (label_dest_dir, image_dest_dir, image_dest_ppm_dir,
            label_dest_ppm_dir, label_dest_bin_dir) = _create_folders(output_dir, 'original')

    # create folder hierarchy for the resized images
    if resize:
        (label_dest_dir_resized, image_dest_dir_resized,
            image_dest_ppm_dir_resized, label_dest_ppm_dir_resized,
            label_dest_bin_dir_resized) = _create_folders(output_dir, '{}x{}'.format(resize[0], resize[1]))
    print("Finished creating directories")

    print("Creating image files")
    for file_name in tqdm.tqdm(os.listdir(image_input_dir)):
        img = skimage.data.load(os.path.join(image_input_dir, file_name))
        file_prefix = file_name[0:file_name.index('.')]

        if not onlyresize:
            skimage.io.imsave(os.path.join(image_dest_dir, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(image_dest_ppm_dir, '{}.ppm'.format(file_prefix)), img)

        if resize:
            img = skimage.transform.resize(img, (resize[1], resize[0]), order=3)
            skimage.io.imsave(os.path.join(image_dest_dir_resized, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(image_dest_ppm_dir_resized, '{}.ppm'.format(file_prefix)), img)

    print("Creating label files")
    dataset = KittiDataset()
    for file_name in tqdm.tqdm(os.listdir(label_input_dir)):
        img = skimage.data.load(os.path.join(label_input_dir, file_name))
        file_prefix = file_name[0:file_name.index('.')]

        if not onlyresize:
            skimage.io.imsave(os.path.join(label_dest_dir, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(label_dest_ppm_dir, '{}.ppm'.format(file_prefix)), img)

            binary_image = np.apply_along_axis(lambda a: dataset.color2label[(a[0], a[1], a[2])].trainId, 2, img)
            binary_image = binary_image.astype(np.uint8)
            io.dump_nparray(binary_image, os.path.join(label_dest_ppm_dir, '{}.bin'.format(file_prefix)))

        if resize:
            img = skimage.transform.resize(img, (resize[1], resize[0]), order=0, preserve_range=True)
            img = img.astype(np.uint8)

            skimage.io.imsave(os.path.join(label_dest_dir_resized, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(label_dest_ppm_dir_resized, '{}.ppm'.format(file_prefix)), img)

            binary_image = np.apply_along_axis(lambda a: dataset.color2label[(a[0], a[1], a[2])].trainId, 2, img)
            binary_image = binary_image.astype(np.uint8)
            io.dump_nparray(binary_image, os.path.join(label_dest_bin_dir_resized, '{}.bin'.format(file_prefix)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates all the necessary files needed'
                                     ' for the semantic segmentation pipeline')
    parser.add_argument('dataset_dir', type=str, help='Root directory for the kitti dataset'
                        ' (contains train, val, train_train and train_val subfolders)')
    parser.add_argument('output_dir', type=str, help='Destination directory')
    parser.add_argument('subset', type=str, help='Subset for which files are generated')
    parser.add_argument('--resize', nargs=2, type=int, help='Width and height of the resized image')
    parser.add_argument('--onlyresize', type=bool, help='Used if only the resized files are going to be generated')

    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.subset, args.resize, args.onlyresize)
