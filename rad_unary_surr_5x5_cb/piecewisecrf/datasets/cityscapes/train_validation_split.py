import argparse
import os
import random
import shutil
import tqdm


def choose_cities(img_input_dir, min_size, max_size):
    '''

    Randomly selects cities from the train dataset such that the
    total number of selected images is between min_size and max_size

    Parameters
    ----------
    img_input_dir : str
        Path to the directory containing training images

    min_size: int
        Minimum allowed size of the validation subset

    max_size: int
        Maximum allowed size of the validation subset

    Returns
    -------
    picked_cities: list
        List of picked image names

    current_size: int
        Number of picked images


    '''
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


def main(img_input_dir, label_input_dir, min_size, max_size):
    '''

    Splits the train dataset into train and validation subsets

    Parameters
    ----------
    img_input_dir : str
        Path to the image folder

    label_input_dir: str
        Path to the labels folder

    min_size: int
        Minimum allowed size of the validation subset

    max_size: int
        Maximum allowed size of the validation subset


    '''
    val_cities, val_size = choose_cities(img_input_dir, min_size, max_size)
    print("Picked cities:", list(val_cities))
    print("Validation set size:", val_size)

    print("Processing images")

    print("Creating train_train and train_val folders")
    train_path = os.path.join(img_input_dir, 'train_train')
    val_path = os.path.join(img_input_dir, 'train_val')
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
    parser = argparse.ArgumentParser(description='Splits train set into train and validation sets.'
                                     ' The validation set contains images taken from specific cities.'
                                     ' This way we have a very good approximation of the test set for '
                                     'validation because the model will be validated on a totally new city context.')
    parser.add_argument('img_input_dir', type=str, help='Image root directory for the cityscapes dataset'
                        ' (contains train, val and test subfolders)')
    parser.add_argument('label_input_dir', type=str, help='Label root directory for the cityscapes dataset'
                        ' (contains train, val and test subfolders)')
    parser.add_argument('min_size', type=int, help='Minimum number of images in the validation set')
    parser.add_argument('max_size', type=int, help='Maximum number of images in the validation set')

    args = parser.parse_args()
    main(args.img_input_dir, args.label_input_dir, args.min_size, args.max_size)
