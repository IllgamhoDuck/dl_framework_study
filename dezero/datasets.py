
import numpy as np
import gzip
import matplotlib.pyplot as plt

from dezero.utils import get_file, cache_dir
from dezero.transforms import Compose, Flatten, ToFloat, Normalize


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

class MNIST(Dataset):
    def __init__(self,
                 train=True,
                 transform=Compose([Flatten(),
                                    ToFloat(),
                                    Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = "http://yann.lecun.com/exdb/mnist/"
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files

        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                # Get random images from MNIST and place it at numpy array
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

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
