import chainer
import argparse
import numpy as np
import seaborn as sns
from pathlib import Path
from chainer import Chain
from chainer import links as L
from chainer import functions as F
from chainer.backends import cuda
from chainer.datasets import TupleDataset
from chainer.training import extensions, make_extension
from chainer.iterators import SerialIterator
from chainer.optimizers import Adam
from chainer.training.trainer import Trainer
from matplotlib import pyplot as plt
from mcmcmll.updater import MCMCMLLUpdater
from mcmcmll.sampler import UniformInitializer, LangevinSampler


class Model(Chain):

    def __init__(self, h=64):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, h)
            self.l2 = L.Linear(h)
            self.l3 = L.Linear(h)
            self.l4 = L.Linear(h)

    def forward(self, x):
        h = self.l1(x)
        h = F.leaky_relu(h)
        h = self.l2(h)
        h = F.leaky_relu(h)
        h = self.l3(h)
        h = F.leaky_relu(h)
        h = self.l4(h)
        return F.sum(h) / x.shape[0]


def plot_sample(sampler):

    @make_extension(trigger=(1, 'epoch'))
    def f(trainer: Trainer):
        epoch = trainer.updater.epoch

        model = trainer.updater._optimizers['main'].target
        with chainer.using_config('train', False):
            neg_batch = sampler.sample(model)
            x = cuda.to_cpu(neg_batch)
            sns.kdeplot(x[:, 0], x[:, 1])
            plt.savefig(f'./{trainer.out}/{epoch}.png')
            plt.close()

    return f


def create_dataset(figpath=None):
    np.random.seed(215)
    n = 1024
    a = np.random.randn(n, 2) + 20*np.ones((n, 2))
    b = np.random.randn(n, 2) - 20*np.ones((n, 2))
    x = np.concatenate([a, b], axis=0)
    np.random.shuffle(x)
    if figpath is not None:
        sns.kdeplot(x[:, 0], x[:, 1])
        plt.savefig(figpath)
        plt.close()
    x = x.astype('f')
    return TupleDataset((x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='result')
    parser.add_argument('--n-epoch', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--device', type=int, default=-1)
    args = parser.parse_args()

    dset = create_dataset(figpath=Path(args.out)/'train.png')
    iterator = SerialIterator(dset, batch_size=512, shuffle=True, repeat=True)
    model = Model()
    opt = Adam(alpha=args.gamma).setup(model)

    initializer = UniformInitializer((512, 2))
    sampler = LangevinSampler(initializer)
    updater = MCMCMLLUpdater(sampler, iterator, opt, device=args.device)
    trainer = Trainer(updater, (args.n_epoch, 'epoch'))
    trainer.extend(plot_sample(sampler))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/logq', 'main/logp']))
    trainer.run()


if __name__ == '__main__':
    main()
