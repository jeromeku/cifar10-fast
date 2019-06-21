import attr
from typing import NamedTuple
import argparse
from functools import partial
import toolz
import torch
from torch import nn

from models import DAWN_net

import core
from core import union, remove_by_type, PiecewiseLinear, Timer
from core import transpose, normalise, pad
from core import TableLoggerV2 as TableLogger
from core import rel_path
from core import Transform, Crop, FlipLR, train_epoch

import torch_backend
from torch_backend import cifar10
from torch_backend import Add, batch_norm, cat, Concat, Correct, Flatten, Identity, Mul, Network
from torch_backend import SGD, trainable_params, Batches

from config import DEFAULT_DATA_PATH, commandline_config
import models
import optimizers
from utils import timed, FileLogger


class ExperimentBuilder:
    def __init__(self, **kwargs):
        self._add_attr(**kwargs)

    def _add_attr(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _load_data(self):
        raise NotImplementedError

    def _preprocess(self, dataset):
        with timed("\nPreprocessing training data"):
            train_data, train_labels = dataset['train']['data'], dataset['train']['labels']
            if self.config.train_preprocessors:
                train_data = toolz.pipe(
                    train_data, *self.config.train_preprocessors)
            train_set = list(zip(train_data, train_labels))

        with timed("Preprocessing test data"):
            test_data, test_labels = dataset['test']['data'], dataset['test']['labels']

            if self.config.test_preprocessors:
                test_data = toolz.pipe(
                    test_data, *self.config.test_preprocessors)
            test_set = list(zip(test_data, test_labels))

        return train_set, test_set

    def _augment(self, dataset):
        with timed("\nAugmenting dataset..."):
            augmented = Transform(dataset, self.config.augmentations)
        return augmented

    def _build_model(self, device=None, batch_size=None):
        with timed(f"\nBuilding model..."):
            net = vars(models)[self.config.net]
            model = Network(union(net(), self.config.losses))
            if device:
                print(f"Transferring model to device gpu:{device}")
                model = model.to(device)
        return model

    def _post_build_process(self, model):
        with timed(f"\nPost processing model..."):
            model = toolz.pipe(model, *self.config.post_build_processors)
        return model

    def _build_schedule(self):
        return self.config.lr_schedule(**self.config.lr_schedule_kwargs)

    def _build_optimizer(self, model, lr):

        return self.config.optimizer(trainable_params(model), lr=lr, **self.config.optimizer_kwargs)

    @classmethod
    def generate(cls, **kwargs):
        print(cls)
        self = cls(**kwargs)
        dataset = self._load_data()
        train, test = self._preprocess(dataset)
        train_augmented = self._augment(train)
        model = self._build_model(device=kwargs.get("device", None))
        model = self._post_build_process(model)
        lr_schedule = self._build_schedule()
        optimizer = self._build_optimizer(model, lr=lr_schedule)

        self._add_attr(model=model, lr_schedule=lr_schedule, optimizer=optimizer, dataset=dataset,
                       train=train, test=test, train_augmented=train_augmented)
        return self

    def __str__(self):
        pass


def load_cifar10(experiment):
    return cifar10(experiment._data_path)


def preprocess(experiment, dataset):
    with timed("Preprocessing training data"):
        train_set = list(zip(transpose(
            normalise(pad(dataset['train']['data'], 4))), dataset['train']['labels']))
    with timed("Preprocessing test data"):
        test_set = list(
            zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))

    return train_set, test_set


@attr.s(kw_only=True)
class ExperimentConfig:
    data_path = attr.ib(validator=attr.validators.instance_of(str))
    net = attr.ib(validator=attr.validators.instance_of(str))
    train_preprocessors = attr.ib(factory=list)
    test_preprocessors = attr.ib(factory=list)
    augmentations = attr.ib(factory=list)
    post_build_processors = attr.ib(factory=list)
    optimizer = attr.ib(default=optimizers.SGD)
    optimizer_kwargs = attr.ib(factory=dict)
    lr_schedule = attr.ib(default=optimizers.ConstLR)
    lr_schedule_kwargs = attr.ib(default={"lr": .1})
    losses = attr.ib(factory=dict)

    def __str__(self):
        format_str = "\n".join(
            [f"{k}: {v}" for k, v in attr.asdict(self).items()])
        return format_str


def to_half(model):
    _ = [m.half() for m in model.children()]
    return model


def to_device(model, device=0):
    return model.to(device)


BaselineConfig = ExperimentConfig(
    data_path='./data', net='DAWN_net',
    train_preprocessors=[toolz.curry(pad)(border=4), normalise, transpose],
    test_preprocessors=[normalise, transpose],
    augmentations=[Crop(32, 32), FlipLR()],
    post_build_processors=[to_half, to_device],
    losses={'loss': (nn.CrossEntropyLoss(reduce=False), [('classifier',), ('target',)]),
              'correct': (Correct(), [('classifier',), ('target',)])},
    optimizer=optimizers.SGD,
    optimizer_kwargs=dict(momentum=0.9, nesterov=True),
    lr_schedule=optimizers.PiecewiseLR,
    lr_schedule_kwargs={"knots": [0, 15, 30, 35], "lrs": [0, 0.1, 0.005, 0]}
)


class BaselineExperiment(ExperimentBuilder):
    config = BaselineConfig

    def _load_data(self):
        return cifar10(self.config.data_path)
