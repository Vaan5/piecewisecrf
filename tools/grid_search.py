import os
import argparse
import subprocess
from multiprocessing.pool import ThreadPool
import pickle

import tools.grid_config as config


def run_crf(exe, dataset, zipped_input, temp_dir, unary_dir, smoothness_theta, smoothness_w,
            appearance_theta_pos, appearance_theta_rgb, appearance_w):
    '''

    Runs a dense crf inference process for the given call arguments

    Parameters
    ----------
    dataset: str
        Dataset name (kitti or cityscapes)

    zipped_input: list
        list of tuples (input_file_name, input_file_path)

    unary_dir: str
        Path to the directory containing unary potentials

    temp_dir: str
        Path to the output directory where images will be saved

    exe: str
        Path to the dense crf executable

    smoothness_theta, smoothness_w, appearance_theta_pos, appearance_theta_rgb, appearance_w: float
        Dense crf kernel parameters

    '''
    for filename, filepath in zipped_input:
        subprocess.call([exe,
                         dataset,
                         filepath,
                         os.path.join(unary_dir, '{}.bin'.format(filename)),
                         os.path.join(temp_dir, '{}.bin'.format(filename)),
                         str(smoothness_theta),
                         str(smoothness_w),
                         str(appearance_theta_pos),
                         str(appearance_theta_rgb),
                         str(appearance_w)])


def evaluate_params(args):
    '''

    Runs crf for each of the provided parameter configurations

    Parameters
    ----------
    args: list
        List of tuples (call arguments for dense crf inference)

    Returns
    -------
    ret_val: tuple
        Crf kernel parameters:
            smoothness theta
            smoothness weight
            appearance theta rgb
            appearance theta pos
            appearance weight


    '''
    (exe, dataset, i, temp_dir, zipped_input, unary_dir, labels_dir,
        smoothness_theta, smoothness_w, appearance_theta_rgb,
        appearance_theta_pos, appearance_w) = args

    temp_dir = os.path.join(temp_dir, 'temp{}'.format(i))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    run_crf(exe, dataset, zipped_input, temp_dir, unary_dir, smoothness_theta,
            smoothness_w, appearance_theta_pos, appearance_theta_rgb, appearance_w)

    return smoothness_theta, smoothness_w, appearance_theta_rgb, appearance_theta_pos, appearance_w


def grid_search(search_ranges, input_dir, unary_dir, labels_dir, temp_dir, exe, dataset, number_of_processes=None):
    '''

    Creates tuples of call arguments needed for each call of dense crf inference.
    Starts the grid search.

    Parameters
    ----------
    search_ranges: dict
        Configuration dictionary for the grid search

    input_dir: str
        Path to the directory containing input images

    unary_dir: str
        Path to the directory containing unary potentials

    labels_dir: str
        Path to the directory containing ground truth files

    temp_dir: str
        Path to the directory where temporary results will be saved

    exe: str
        Path to the dense crf executable

    dataset: str
        kitti or cityscapes

    number_of_processes: int
        Number of processes to create when running grid search


    '''
    # prepare input files, labels and temporary directories
    input_filenames = os.listdir(input_dir)
    input_filepaths = [os.path.join(input_dir, name) for name in input_filenames]

    input_filenames = [x[0:x.find('.ppm')] for x in input_filenames]

    # optimize appearance kernel parameters
    params = []
    counter = 0
    for smoothness_theta in search_ranges['smoothness_theta']:
        for smoothness_w in search_ranges['smoothness_w']:
            for appearance_theta_rgb in search_ranges['appearance_theta_rgb']:
                for appearance_theta_pos in search_ranges['appearance_theta_pos']:
                    for appearance_w in search_ranges['appearance_w']:
                        params.append((exe, dataset, counter, temp_dir, zip(input_filenames, input_filepaths),
                                       unary_dir, labels_dir, smoothness_theta, smoothness_w,
                                       appearance_theta_rgb, appearance_theta_pos, appearance_w))
                        counter += 1

    results = ThreadPool(number_of_processes).map(evaluate_params, params)


if __name__ == '__main__':

    search_ranges = config.search_ranges

    parser = argparse.ArgumentParser(description="Runs a grid search for estimating optimal dense crf parameters")
    parser.add_argument('input_dir', type=str, help="Input images directory")
    parser.add_argument('unary_dir', type=str, help="Directory containing unary potentials")
    parser.add_argument('labels_dir', type=str, help="Directory containing label files")
    parser.add_argument('temp_dir', type=str, help="Directory in which temporary results will be saved")
    parser.add_argument('exe', type=str, help="Path to dense crf executable")
    parser.add_argument('dataset', type=str, help="cityscapes or kitti")
    parser.add_argument('--processes', nargs=1, type=int, help="Number of processes in the pool")

    args = parser.parse_args()

    number_of_processes = None
    if args.processes:
        number_of_processes = args.processes[0]

    grid_search(search_ranges, args.input_dir, args.unary_dir, args.labels_dir,
                args.temp_dir, args.exe, args.dataset, number_of_processes)
