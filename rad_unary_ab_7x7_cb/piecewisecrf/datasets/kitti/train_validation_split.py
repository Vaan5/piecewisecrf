import argparse
import os
import random
import shutil
import tqdm

import numpy as np

def choose_cities(img_input_dir, min_size, max_size):
    cities = next(os.walk(os.path.join(img_input_dir, 'train')))[1]

    current_size = 0
    picked_cities = set()
    while current_size < min_size:
        picked_city = random.choice(cities)
        if picked_city not in picked_cities:
            picked_cities.add(picked_city)
            number_of_images = len(next(os.walk(os.path.join(os.path.join(img_input_dir, 'train'), picked_city)))[2])
            if current_size + number_of_images > max_size:
                picked_cities.remove(picked_city)
            else:
                current_size += number_of_images

    return picked_cities, current_size


def _pick_images(labels_dir, size):
    images = os.listdir(labels_dir)
    return random.sample(images, size)


def _label_statistics(image_paths):
    '''

    Calculates label statistics (number of picked pixels for each class)

    Parameters
    ----------
    image_paths : list
        List of absolute paths for picked images


    '''
    for i in image_paths:




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
        stats = _label_statistics([os.path.join(img_input_dir, x) for x in picked_images])

    val_cities, val_size = choose_cities(img_input_dir, min_size, max_size)
    print("Picked cities:", list(val_cities))
    print("Validation set size:", val_size)






    print("Processing images")
    print("Creating train_train and train_val folders")
    train_path = os.path.join(dataset_dir, 'train_train')
    val_path = os.path.join(dataset_dir, 'train_val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    print("Finished creating folders")





    print("Moving image files")
    src_dir = os.path.join(img_input_dir, 'train')
    for city in tqdm.tqdm(next(os.walk(src_dir))[1]):
        if city in val_cities:
            city_val_path = os.path.join(val_path, city)
            if not os.path.exists(city_val_path):
                os.makedirs(city_val_path)

            print("Moving {}".format(city))
            src_path = os.path.join(src_dir, city)
            for img in tqdm.tqdm(next(os.walk(src_path))[2]):
                shutil.copy(os.path.join(src_path, img), os.path.join(city_val_path, img))
        else:
            city_train_path = os.path.join(train_path, city)
            if not os.path.exists(city_train_path):
                os.makedirs(city_train_path)

            print("Moving {}".format(city))
            src_path = os.path.join(src_dir, city)
            for img in tqdm.tqdm(next(os.walk(src_path))[2]):
                shutil.copy(os.path.join(src_path, img), os.path.join(city_train_path, img))

    print("Processing labels")

    print("Creating train_train and train_val folders")
    train_path = os.path.join(label_input_dir, 'train_train')
    val_path = os.path.join(label_input_dir, 'train_val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    print("Finished creating folders")

    print("Moving image files")
    src_dir = os.path.join(label_input_dir, 'train')
    for city in tqdm.tqdm(next(os.walk(src_dir))[1]):
        if city in val_cities:
            city_val_path = os.path.join(val_path, city)
            if not os.path.exists(city_val_path):
                os.makedirs(city_val_path)

            print("Moving {}".format(city))
            src_path = os.path.join(src_dir, city)
            for img in tqdm.tqdm(next(os.walk(src_path))[2]):
                shutil.copy(os.path.join(src_path, img), os.path.join(city_val_path, img))
        else:
            city_train_path = os.path.join(train_path, city)
            if not os.path.exists(city_train_path):
                os.makedirs(city_train_path)

            print("Moving {}".format(city))
            src_path = os.path.join(src_dir, city)
            for img in tqdm.tqdm(next(os.walk(src_path))[2]):
                shutil.copy(os.path.join(src_path, img), os.path.join(city_train_path, img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits train set into train and validation sets. '
                                     'Ensures that the validation set contains the same distribution of labels')
    parser.add_argument('dataset_path', type=str, help='Path to the root directory of the kitti dataset'
                        ' (contains train and valid subdirectories)')
    parser.add_argument('size', type=int, help='Number of images in the validation set')

    args = parser.parse_args()
    main(args.dataset_path, args.size)
