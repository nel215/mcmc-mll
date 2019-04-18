import chainer
from chainer.training.updaters import StandardUpdater


class MCMCMLLUpdater(StandardUpdater):

    def __init__(
        self,
        sampler,
        *args,
        **kwargs,
    ):
        super(MCMCMLLUpdater, self).__init__(*args, **kwargs)
        self.sampler = sampler

    def update_core(self):
        iterator = self._iterators['main']
        pos_batch = iterator.next()
        pos_batch = self.converter(pos_batch, self.device)
        pos_batch = pos_batch[0]

        optimizer = self._optimizers['main']
        model = optimizer.target
        neg_batch = self.sampler.sample(model)

        logq = model(pos_batch)
        logp = model(neg_batch)
        loss = logq - logp
        chainer.report({
            'loss': loss,
            'logq': logq,
            'logp': logp,
        }, model)

        optimizer.update(lambda: loss)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
