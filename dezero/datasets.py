import numpy as np

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
