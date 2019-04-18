import chainer
import numpy as np
from chainer import Parameter


class UniformInitializer:

    def __init__(self, shape):
        self.shape = shape

    def initialize(self):
        return np.random.uniform(size=self.shape).astype('f')


class LangevinSampler:
    def __init__(
        self,
        initializer,
        n_samples=None,
        eps=None,
        tau=None,
        step=None,
    ):
        self.initializer = initializer
        self.eps = eps or 1e-1
        self.tau = tau or 1
        self.step = step or 50

    def sample(self, model):
        xp = model.xp
        eps = self.eps
        tau = self.tau

        batch = self.initializer.initialize()
        batch = Parameter(xp.asarray(batch))
        for i in range(self.step):
            loss = model(batch)
            grad, = chainer.grad([loss], [batch])
            z = xp.random.randn(*batch.shape)
            batch = batch - eps*eps*0.5*grad + eps*tau*z
        return batch.data
