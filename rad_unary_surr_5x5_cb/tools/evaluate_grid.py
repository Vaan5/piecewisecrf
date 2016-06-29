import os
import argparse
import calculate_accuracy_t

import tools.grid_config as config


if __name__ == '__main__':

    search_ranges = config.search_ranges

    params = []
    counter = 0
    for smoothness_theta in search_ranges['smoothness_theta']:
        for smoothness_w in search_ranges['smoothness_w']:
            for appearance_theta_rgb in search_ranges['appearance_theta_rgb']:
                for appearance_theta_pos in search_ranges['appearance_theta_pos']:
                    for appearance_w in search_ranges['appearance_w']:
                        params.append((counter, smoothness_theta, smoothness_w,
                                       appearance_theta_rgb, appearance_theta_pos,
                                       appearance_w))
                        counter += 1

    for i in params:
        print(i)

    parser = argparse.ArgumentParser(description="Evaluates results of grid_search call. "
                                     "This is done separately for easier remote evaluation "
                                     "(in case of redirecting output to file) - synchronized print")
    parser.add_argument('input_dir', type=str, help="Path to the temp_dir from grid_search")
    parser.add_argument('labels_dir', type=str, help="Path to labels directory")

    args = parser.parse_args()

    with open('evaluation.txt', "w") as ff:
        best_iou = 0
        best_dir = 0
        for directory in os.listdir(args.input_dir):
            if len(os.listdir(os.path.join(args.input_dir, directory))) == 0:
                continue
            ff.write(directory + "\n")
            print(directory)
            r = calculate_accuracy_t.evaluate_segmentation(os.path.join(args.input_dir,
                                                                        directory), args.labels_dir)
            ff.write("#{} => {} - {}\n".format(directory, r[5], r[3]))
            print("#{} => {} - {}\n".format(directory, r[5], r[3]))
            if r[5] > best_iou:
                best_iou = r[5]
                best_dir = directory

        print("Best IOU", best_iou)
        print(best_dir)

        ff.write("Best IOU: {}\nBest dir: {}".format(best_iou, best_dir))
