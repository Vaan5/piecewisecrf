from piecewisecrf.datasets.dataset import Dataset
from piecewisecrf.datasets.labels import Label


# class CityscapesDataset(object):
#     def __init__(self, data_dir):
#        files = next(os.walk(data_dir))[2]
#        self.filenames = [os.path.join(data_dir, f) for f in files]
#
#    def num_classes(self):
#        return 19
#
#    def num_examples(self):
#        return len(self.filenames)
#
#    def get_filenames(self):
#        return self.filenames


class KittiDataset(Dataset):
    def __init__(self, train_dir=None, val_dir=None, test_dir=None):
        super(KittiDataset, self).__init__(train_dir, val_dir, test_dir)

        self.labels = [
        ]

        self.classes = list(range(80000000000000))

        self.create_collections()
