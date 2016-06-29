import os


class Dataset(object):
    '''

    Class used for encapsulating all dataset information.
    Implementing classes need to fill the following collections:
        - labels
        - classes
    At the end of the constructor a call to create_collections is necessary.


    '''
    def __init__(self, train_dir=None, val_dir=None, test_dir=None):
        self.train_filenames = []
        self.validation_filenames = []
        self.test_filenames = []

        # implementing classes need to fill these collections
        self.labels = []
        self.classes = []
        self.name2label = {}
        self.id2label = {}
        self.trainId2label = {}
        self.color2label = {}
        self.category2labels = {}

        if train_dir:
            files = next(os.walk(train_dir))[2]
            self.train_filenames = [os.path.join(train_dir, f) for f in files]

        if val_dir:
            files = next(os.walk(val_dir))[2]
            self.validation_filenames = [os.path.join(val_dir, f) for f in files]

        if test_dir:
            files = next(os.walk(test_dir))[2]
            self.test_filenames = [os.path.join(test_dir, f) for f in files]

    def get_filenames(self, partition='train'):
        if partition == 'train':
            return self.train_filenames
        elif partition == 'validation':
            return self.validation_filenames
        elif partition == 'test':
            return self.test_filenames

    def num_examples(self, partition='train'):
        if partition == 'train':
            return len(self.train_filenames)
        elif partition == 'validation':
            return len(self.validation_filenames)
        elif partition == 'test':
            return len(self.test_filenames)

    def num_classes(self):
        return len(self.classes)

    def create_collections(self):
        self.name2label = {label.name: label for label in self.labels}
        self.id2label = {label.id: label for label in self.labels}
        self.trainId2label = {label.trainId: label for label in reversed(self.labels)}
        self.color2label = {label.color: label for label in reversed(self.labels)}
        self.category2labels = {}
        for label in self.labels:
            category = label.category
            if category in self.category2labels:
                self.category2labels[category].append(label)
            else:
                self.category2labels[category] = [label]
