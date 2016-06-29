import os
import argparse
import subprocess
from multiprocessing.pool import ThreadPool


def run_crf(args):
    '''

    Runs a dense crf inference process for the given call arguments

    '''
    (dataset, exe, filename, filepath, unary_dir, output_dir,
        smoothness_theta, smoothness_w, appearance_theta_pos,
        appearance_theta_rgb, appearance_w) = args
    subprocess.call([exe,
                     dataset,
                     filepath,
                     os.path.join(unary_dir, '{}.bin'.format(filename)),
                     os.path.join(output_dir, '{}.bin'.format(filename)),
                     str(smoothness_theta),
                     str(smoothness_w),
                     str(appearance_theta_pos),
                     str(appearance_theta_rgb),
                     str(appearance_w)])


def generate_data(dataset, input_dir, unary_dir, output_dir, exe, smoothness_theta, smoothness_w,
                  appearance_theta_pos, appearance_theta_rgb, appearance_w):
    '''

    Generates call arguments for dense crf inference

    Parameters
    ----------
    dataset: str
        Dataset name (kitti or cityscapes)

    input_dir: str
        Path to the directory containing input images

    unary_dir: str
        Path to the directory containing unary potentials

    output_dir: str
        Path to the output directory where images will be saved

    exe: str
        Path to the dense crf executable

    smoothness_theta, smoothness_w, appearance_theta_pos, appearance_theta_rgb, appearance_w: float
        Dense crf kernel parameters

    Returns
    -------
    list: list
        List of tuples (call arguments)

    '''
    input_filenames = os.listdir(input_dir)

    input_filenames = [x[0:x.find('.ppm')] for x in input_filenames]

    return [(dataset, exe, name, os.path.join(input_dir, "{}.ppm".format(name)),
             unary_dir, output_dir, smoothness_theta, smoothness_w,
             appearance_theta_pos, appearance_theta_rgb, appearance_w) for name in input_filenames]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script used for generating dense crf output for the given image directory")
    parser.add_argument('dataset', type=str, help="cityscapes or kitti")
    parser.add_argument('input_dir', type=str,
                        help="Directory containing input images. They must be in ppm format")
    parser.add_argument('unary_dir', type=str,
                        help="Directory containing unary potentials. They must be in binary format")
    parser.add_argument('output_dir', type=str,
                        help="Destination directory in which the results will be saved (binary format)")
    parser.add_argument('exe', type=str, help="Path to dense crf executable")
    parser.add_argument('smoothness_theta', type=str, help="Deviation parameter for the smoothness kernel")
    parser.add_argument('smoothness_w', type=str, help="Smoothness kernel weight")
    parser.add_argument('appearance_theta_pos', type=str,
                        help="Position deviation parameter for the appearance kernel")
    parser.add_argument('appearance_theta_rgb', type=str,
                        help="Color deviation parameter for the appearance kernel")
    parser.add_argument('appearance_w', type=str, help="Appearance kernel weight")
    parser.add_argument('--processes', nargs=1, type=int, help="Number of processes in the pool")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data = generate_data(args.dataset, args.input_dir, args.unary_dir, args.output_dir, args.exe,
                         args.smoothness_theta, args.smoothness_w, args.appearance_theta_pos,
                         args.appearance_theta_rgb, args.appearance_w)

    number_of_processes = None
    if args.processes:
        number_of_processes = args.processes[0]
    results = ThreadPool(number_of_processes).map(run_crf, data)
