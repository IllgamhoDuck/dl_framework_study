import numpy as np

import dezero
from dezero import cuda
from dezero import Variable
from dezero import Function
from dezero.core import as_array
from dezero.core import as_variable


# Neural Network

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


# Calculation

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * cos(x)
        return gx

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y**2)
        return gx

class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def sin(x):
    return sin()(x)

def cos(x):
    return cos()(x)

def tanh(x):
    return tanh()(x)

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)


# Shape transform

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            gx = transpose(gy)
        else:
            axes_len = len(self.axes)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
            gx = transpose(gy, inv_axes)
        return gx

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)


# slicing

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = GetItemGrad(self.slices, x.shape)(gy)
        return gx

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


# Broadcast

class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = dezero.utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = dezero.utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


# Matrix

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)


# Regularization

class Dropout(Function):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, x):
        if dezero.Config.train:
            xp = cuda.get_array_module(x)

            mask = xp.random.rand(*x.shape) > self.p
            scale = xp.array(1.0 - self.p).astype(x.dtype)

            self.mask = mask
            self.scale = scale

            y = x * mask / scale
        else:
            y = x

        return y

    def backward(self, gy):
        if dezero.Config.train:
            gx = (gy / self.scale) * self.mask
        else:
            gx = gy

        return gx

def dropout(x, p=0.5):
    return Dropout(p=p)(x)


# activation

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + np.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def sigmoid(x):
    return Sigmoid()(x)

def relu(x):
    return ReLU()(x)


# Loss

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx = gx - (y * sumdx)
        return gx

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_p = log_softmax(x)
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy = gy / N

        # p_i
        y = softmax(x)

        # t_i
        xp = cuda.get_array_module(y.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]

        # p_i - t_i
        gx = (y - t_onehot) * gy

        return gx

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def softmax(x, axis=1):
    return Softmax(axis)(x)

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def log_softmax(x, axis=1):
    xp = cuda.get_array_module(x)
    x_max = x.max(axis=axis, keepdims=True)
    y = x - x_max

    y = xp.exp(y)
    y_sum = y.sum(axis=axis, keepdims=True)

    log_y_sum = xp.log(y_sum)
    log_z = x_max + log_y_sum

    log_p = x - log_z

    return log_p


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


# Metrics

def accuracy(y_pred, y_true):
    y_pred, y_true = as_variable(y_pred), as_variable(y_true)

    y_pred = y_pred.data.argmax(axis=1).reshape(y_true.shape)
    result = (y_pred == y_true.data)
    acc = result.mean()

    return Variable(as_array(acc))


# simplified versions

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b in None:
        return t

    y = t + b

    # t is only used for back propagation
    # The value inside doesn't matter so we can remove it
    # to save memory
    t.data = None
    return y

def dropout_simple(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)

        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


# Max / Mix / Clip

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x, self.axis)
        gy = gy.reshape(shape)
        y = y.reshape(shape)

        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)

        gx = gy * cond

        return gx

class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)
