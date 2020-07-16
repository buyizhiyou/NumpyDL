# -*- coding: utf-8 -*-


import numpy as np

from .base import Layer
from .. import initializations


class BatchNormal(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014) [1]_ .

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Parameters
    ----------
    epsilon ： small float > 0
        Fuzz parameter. npdl expects epsilon >= 1e-5.
    axis : integer
        axis along which to normalize in mode 0. For instance,
        if your input tensor has shape (samples, channels, rows, cols),
        set axis to 1 to normalize per feature map (channels axis).
    momentum : float
        momentum in the computation of the
        exponential average of the mean and standard deviation
        of the data, for feature-wise normalization.
    beta_init : npdl.initializations.Initializer
        name of initialization function for shift parameter, or alternatively,
        npdl function to use for weights initialization.
    gamma_init : npdl.initializations.Initializer
        name of initialization function for scale parameter, or alternatively,
        npdl function to use for weights initialization.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    References
    ----------
    .. [1] [Batch Normalization: Accelerating Deep Network Training
          by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self, epsilon=1e-6, momentum=0.9, axis=1,
                 beta_init='zero', gamma_init='one'):
        self.epsilon = epsilon
        self.momentum = momentum
        ax = list(np.arange(4))
        if axis is not None:
            ax.pop(axis)
        self.axis = tuple(ax)

        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)

        self.beta, self.dbeta = None, None
        self.gamma, self.dgamma = None, None
        self.cache = None

        self.out_shape = None

    def connect_to(self, prev_layer):
        n_in = prev_layer.out_shape[1]  # N C H W

        self.beta = self.beta_init(((1, n_in, 1, 1)))
        self.gamma = self.gamma_init((1, n_in, 1, 1))

        self.out_shape = prev_layer.out_shape

    def forward(self, input, *args, **kwargs):
        # N, D = x.shape
        # # 为了后向传播求导方便，这里都是分步进行的

        # step1: calculate the mean
        # mu = 1. / N * np.sum(x, axis=0)
        mean = np.mean(input, axis=(0, 1, 2))  # C

        # step2: 减均值
        xmu = input - mean

        # step3: 计算方差
        # sq = xmu ** 2
        # var = 1. / N * np.sum(sq, axis=0)
        var = np.std(xmu, axis=(0, 1, 2))  # C

        # step4: 计算x^的分母项
        sqrtvar = np.sqrt(var + self.epsilon)
        ivar = 1. / sqrtvar

        # step5: normalization->x^
        xhat = xmu * ivar

        # step6: scale and shift
        gammax = self.gamma * xhat
        out = gammax + self.beta

        # 存储中间变量
        self.cache = (xhat, xmu, ivar, sqrtvar, var)

        return out

    def backward(self, pre_grad, *args, **kwargs):
        '''
        https://zhuanlan.zhihu.com/p/45614576
        '''
        # 解压中间变量
        xhat, xmu, ivar, sqrtvar, var = self.cache
        N, D, H, W = pre_grad.shape

        # 当前层要更新的梯度
        self.dbeta = np.sum(pre_grad, axis=self.axis, keepdims=True)
        dgammax = pre_grad
        self.dgamma = np.sum(dgammax * xhat, axis=self.axis, keepdims=True)
        dxhat = pre_grad * self.gamma

        # 回传的梯度
        dx_ = self.gamma * pre_grad
        dx = N * dx_ - np.sum(dx_, axis=self.axis, keepdims=True) - \
            xhat * np.sum(dx_ * xhat, axis=self.axis, keepdims=True)
        dx *= (1.0/N) / np.sqrt(var+self.epsilon)

        return dx

    @property
    def params(self):
        return self.beta, self.gamma

    @property
    def grads(self):
        return self.dbeta, self.dgamma
