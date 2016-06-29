import argparse
import os
import tqdm

import skimage
import skimage.data
import skimage.transform

import numpy as np

import piecewisecrf.datasets.cityscapes as dataset
import piecewisecrf.helpers.io as io


def _create_folder(root_dir, sub_dir):
    dir_path = os.path.join(root_dir, sub_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def _create_folders(root_dir, sub_dir):
    # create folder hierarchy for original size images
    output_dir = _create_folder(root_dir, sub_dir)

    label_dest_dir = _create_folder(output_dir, 'gt')
    image_dest_dir = _create_folder(output_dir, 'img')
    image_dest_ppm_dir = _create_folder(output_dir, 'img_ppm')
    label_dest_ppm_dir = _create_folder(output_dir, 'gt_ppm')
    label_dest_bin_dir = _create_folder(output_dir, 'gt_bin')

    return label_dest_dir, image_dest_dir, image_dest_ppm_dir, label_dest_ppm_dir, label_dest_bin_dir


def main(img_input_dir, label_input_dir, output_dir, subset, resize,
         onlyresize, containsl, replacel, replacei):
    print("Creating directories")
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
    image_input_dir = os.path.join(img_input_dir, subset)
    label_input_dir = os.path.join(label_input_dir, subset)

    cities = next(os.walk(image_input_dir))[1]
    files = []
    for city in cities:
        image_city_dir = os.path.join(image_input_dir, city)
        files.extend([(os.path.join(image_city_dir, x), x) for x in next(os.walk(image_city_dir))[2]])

    for file_path, file_name in tqdm.tqdm(files):
        img = skimage.data.load(file_path)
        file_prefix = file_name[0:file_name.index('.')]
        if replacei:
            file_prefix = file_prefix.replace(replacei[0], replacei[1])

        if not onlyresize:
            skimage.io.imsave(os.path.join(image_dest_dir, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(image_dest_ppm_dir, '{}.ppm'.format(file_prefix)), img)

        if resize:
            img = skimage.transform.resize(img, (resize[1], resize[0]), order=3)
            skimage.io.imsave(os.path.join(image_dest_dir_resized, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(image_dest_ppm_dir_resized, '{}.ppm'.format(file_prefix)), img)

    print("Creating label files")
    cities = next(os.walk(label_input_dir))[1]
    files = []
    for city in cities:
        label_city_dir = os.path.join(label_input_dir, city)
        if containsl:
            files.extend([(os.path.join(label_city_dir, x), x) for x in next(os.walk(label_city_dir))[2]
                          if containsl[0] in x])
        else:
            files.extend([(os.path.join(label_city_dir, x), x) for x in next(os.walk(label_city_dir))[2]])

    for file_path, file_name in tqdm.tqdm(files):
        img = skimage.data.load(file_path)
        file_prefix = file_name[0:file_name.index('.')]
        if replacel:
            file_prefix = file_prefix.replace(replacel[0], replacel[1])

        if not onlyresize:
            skimage.io.imsave(os.path.join(label_dest_dir, '{}.png'.format(file_prefix)), img)
            skimage.io.imsave(os.path.join(label_dest_ppm_dir, '{}.ppm'.format(file_prefix)), img)

            binary_image = np.apply_along_axis(lambda a: dataset.color2label[(a[0], a[1], a[2])].trainId, 2, img)
            binary_image = binary_image.astype(np.uint8)
            io.dump_nparray(binary_image, os.path.join(label_dest_bin_dir, '{}.bin'.format(file_prefix)))

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
    parser.add_argument('img_input_dir', type=str, help='Image root directory for the cityscapes dataset'
                        ' (contains train, val, train_train, train_val and test subfolders)')
    parser.add_argument('label_input_dir', type=str, help='Label root directory for the cityscapes dataset'
                        ' (contains train, val, train_train, trian_val and test subfolders)')
    parser.add_argument('output_dir', type=str, help='Destination directory')
    parser.add_argument('subset', type=str, help='Subset for which files are generated')
    parser.add_argument('--resize', nargs=2, type=int, help='Width and height of the resized image')
    parser.add_argument('--onlyresize', dest='onlyresize', action='store_true', help='Used if only the resized files are going to be generated')
    parser.add_argument('--containsl', nargs=1, type=str, help='String that each label file contains '
                        '(useful if label_input_dir is not homogenous)')
    parser.add_argument('--replacel', nargs=2, type=str, help='String (first argument) in label file name '
                        'which will be replaced with the other argument')
    parser.add_argument('--replacei', nargs=2, type=str, help='String (first argument) in image file name '
                        'which will be replaced with the other argument')

    args = parser.parse_args()
    main(args.img_input_dir, args.label_input_dir, args.output_dir, args.subset,
         args.resize, args.onlyresize, args.containsl, args.replacel, args.replacei)
