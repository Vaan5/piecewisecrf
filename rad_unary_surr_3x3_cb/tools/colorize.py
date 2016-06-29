import argparse
from PIL import Image
import numpy as np
import os

import piecewisecrf.helpers.io as io


def main(input_, output_, dataset):
    '''

    Transforms images in input_ from binary format to ppm images

    Parameters
    ----------
    input_: str
        Path to the input directory

    output_: str
        Path to the output directory

    dataset: Dataset
        kitti or cityscapes


    '''
    if not os.path.exists(output_):
            os.makedirs(output_)

    for file in os.listdir(input_):
        if file.endswith('.bin'):
            a = io.load_nparray_from_bin_file(os.path.join(input_, file), np.int16)
            img = np.zeros((a.shape[0], a.shape[1], 3))

            for i in xrange(a.shape[0]):
                for j in xrange(a.shape[1]):
                    img[i, j, :] = dataset.labels.trainId2label[a[i, j]].color

            img = Image.fromarray(img.astype(np.uint8))
            img.save(os.path.join(output_, file[0:-3] + 'ppm'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts binary label images into color images")
    parser.add_argument('input', help="path to the input directory")
    parser.add_argument('output', help="path to the output directory")
    possible_datasets = ['cityscapes', 'kitti']
    parser.add_argument('dataset_name', type=str, choices=possible_datasets,
                        help='Used dataset')
    args = parser.parse_args()

    if args.dataset_name == possible_datasets[0]:
        from piecewisecrf.datasets.cityscapes.cityscapes import CityscapesDataset
        dataset = CityscapesDataset()
    elif args.dataset_name == possible_datasets[1]:
        from piecewisecrf.datasets.kitti.kitti import KittiDataset
        dataset = KittiDataset()

    main(args.input, args.output, dataset)
