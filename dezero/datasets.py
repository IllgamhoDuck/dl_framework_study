
import numpy as np


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train

        # Transformation
        self.transform = transform
        self.target_transform  = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    # num_data : number of data per class
    num_data, num_class, input_dim = 100, 3, 2

    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    y = np.zeros(data_size, dtype=np.int)

    for class_i in range(num_class):
        for data_i in range(num_data):

            rate = data_i / num_data

            radius = 1.0 * rate
            theta = 4.0*class_i + 4.0*rate + 0.2*np.random.randn()

            ix = num_data*class_i + data_i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()

            y[ix] = class_i

    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    y = y[indices]

    return x, y
