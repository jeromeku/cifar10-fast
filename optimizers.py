import numpy as np
import torch
from torch import nn
import torchvision
from core import build_graph, cat, to_numpy


class TorchOptimiser():
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())

    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k, v in self.opt_params.items()}

    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)


def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum,
                          weight_decay=weight_decay, dampening=dampening,
                          nesterov=nesterov)


class LRScheduler():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _lr_step(self, step):
        raise NotImplementedError

    def __call__(self, step):
        return self._lr_step(step)


class ConstLR(LRScheduler):
    def __init__(self, lr):
        super().__init__(lr=lr)

    def _lr_step(self, step):
        return self.lr


class PiecewiseLR(LRScheduler):
    def __init__(self, knots, lrs):
        super(PiecewiseLR, self).__init__(knots=knots, lrs=lrs)

    def _lr_step(self, step):
        return np.interp([step], self.knots, self.lrs)[0]
