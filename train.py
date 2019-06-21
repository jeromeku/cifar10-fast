'''
TODO
- Refactor train script
- Command line args
- Distributed Data Parallel
- Add mixed precision
- Data prefetching
- Fast collate

'''
from typing import NamedTuple
import argparse
import attr
import datetime
import json
from functools import partial
import logging
import traceback

from models import DAWN_net

import torch
from torch import nn

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
from experiment import ExperimentBuilder, ExperimentConfig, BaselineExperiment, BaselineConfig
from utils import timed, LogBuilder
torch.backends.cudnn.benchmark = True

# Change for distributed


def main(args):
    device = torch.device(config.local_rank)
    logger = LogBuilder(name=__name__).addFileHandler(output_dir=args.log_dir, filename=datetime.datetime.now(
    ).strftime("%Y-%m-%d_%H-%M"), level=logging.INFO).get_logger()
    logger.info("Starting experiment".upper())
    experiment = BaselineExperiment.generate(batch_size=args.batch_size)
    logger.info("ExperimentConfig", extra=attr.asdict(experiment.config))

    try:
        summary = train(experiment, args.batch_size,
                        logger, num_workers=args.num_workers, num_epochs=args.num_epochs)
    except:
        logger.error("uncaught exception: %s", traceback.format_exc())
        raise


def train(experiment, batch_size, logger, num_workers=0, num_epochs=None):
    train_set, test_set = experiment.train_augmented, experiment.test
    model = experiment.model
    lr_schedule = experiment.lr_schedule
    optimizer = experiment.optimizer

    train_batches = Batches(train_set, batch_size, shuffle=True,
                            set_random_choices=True, num_workers=num_workers)
    test_batches = Batches(test_set, batch_size,
                           shuffle=False, num_workers=num_workers)
    num_epochs = num_epochs or lr_schedule.knots[-1]
    optimizer.opt_params['lr'] = lambda step: lr_schedule(
        step/len(train_batches))/batch_size

    table, timer = TableLogger(), Timer()
    for epoch in range(num_epochs):
        epoch_stats = train_epoch(
            model, train_batches, test_batches, optimizer.step, timer, test_time_in_total=True)
        summary = union(
            {'epoch': epoch+1, 'lr': lr_schedule(epoch+1)}, epoch_stats)
        table.append(summary)
        logger.info("training", extra=summary)
    # table.save()
    logger.info("Finished Experiment".upper())
    logger.info(f"training_summary", extra={
                'training_summary_json': json.dumps(table.summaries)})

    return summary


if __name__ == "__main__":
    config = commandline_config()
    main(config)
