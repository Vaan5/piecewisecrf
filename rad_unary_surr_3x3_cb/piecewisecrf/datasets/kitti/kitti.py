from piecewisecrf.datasets.dataset import Dataset
from piecewisecrf.datasets.labels import Label


class KittiDataset(Dataset):
    '''

    Kitti dataset abstraction

    '''
    def __init__(self, train_dir=None, val_dir=None, test_dir=None):
        super(KittiDataset, self).__init__(train_dir, val_dir, test_dir)

        self.labels = [
            # name id trainId category catId hasInstances ignoreInEval color
            Label('sky', 0, 0, 'void', 0, False, False, (128, 128, 128)),
            Label('building', 1, 1, 'void', 0, False, False, (128, 0, 0)),
            Label('road', 2, 2, 'void', 0, False, False, (128, 64, 128)),
            Label('sidewalk', 3, 3, 'void', 0, False, False, (0, 0, 192)),
            Label('fence', 4, 4, 'void', 0, False, False, (64, 64, 128)),
            Label('vegetation', 5, 5, 'void', 0, False, False, (128, 128, 0)),
            Label('pole', 6, 6, 'void', 0, False, False, (192, 192, 128)),
            Label('car', 7, 7, 'void', 0, False, False, (64, 0, 128)),
            Label('sign', 8, 8, 'void', 0, False, False, (192, 128, 128)),
            Label('pedestrian', 9, 9, 'void', 0, False, False, (64, 64, 0)),
            Label('cyclist', 10, 10, 'void', 0, False, False, (0, 128, 192)),
            Label('ignore', 11, 11, 'void', 0, False, True, (0, 0, 0)),
        ]

        self.classes = list(range(11))

        self.create_collections()
