
import math
import numpy as np

from dezero import cuda

class Optimizer:
    """Update the Parameters

    Args:
        target Union[Model, Layer]: The class which has parameter
        hooks Iterable[function]: Preprocessing the parameter
                                  Used to implement like Weight Decay or Gradient Clipping
    """
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data

        param.data += v

class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        h = self.hs[h_key]
        grad = param.grad.data

        h += grad * grad
        param.data -= self.lr * grad / (xp.sqrt(h) + self.eps)

class RMSProp(Optimizer):
    def __init__(self, lr=0.001, rho=0.95, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.msg = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)

        msg = self.msg[key]
        grad = param.grad.data

        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        param.data -= self.lr * grad / (xp.sqrt(msg) + self.eps)

class AdaDelta(Optimizer):
    def __init__(self, lr=0.001, rho=0.95, eps=1e-6):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)
            self.msdx[key] += self.lr

        msg, msdx = self.msg[key], self.msdx[key]
        grad = param.grad.data

        msg *= self.rho
        msg += (1 - self.rho) * grad * grad

        dx = xp.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx

        param.data -= dx

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m *= beta1
        m += (1 - beta1) * grad

        v *= beta2
        v += (1 - beta2) * grad * grad

        m_hat = m / (1 - beta1)
        v_hat = v / (1 - beta2)

        param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + eps)


class WeightDecay:
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data

class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, params):
        # norm is a size
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate

class FreezeParam:
    def __init__(self, *layers):
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params):
        for p in self.freeze_params:
            p.grad = None

class FreezePytorch:
    def __call__(self, params):
        for param in params:
            if param.requires_grad == False:
                param.grad = None
