import json
import logging
import os
import pandas as pd
from contextlib import contextmanager
from core import Timer

from pythonjsonlogger import jsonlogger
import sys


@contextmanager
def timed(enter_msg=None, exit_msg=None):
    t = Timer()
    try:
        # t = timer()
        if enter_msg:
            print(enter_msg)
        yield
    finally:
        elapsed = t()
        print(f'Finished in {elapsed: .2} seconds')
        if exit_msg:
            print(exit_msg)


class LoggingFormats():
    MSG = '%(message)s'
    TIME = '%(levelname)s %(asctime)s - %(name)s:%(lineno)d - %(message)s'


class LoggingHandlerBuilder():

    @staticmethod
    def configHandler(handler, level, format, formatter=logging.Formatter, logger=None, attach=True):
        handler.setLevel(level)
        handler.setFormatter(formatter(format))
        if attach:
            assert logger
            logger.addHandler(handler)
        return handler

    @staticmethod
    def createFileHandler(fn, level, format, formatter=logging.Formatter, logger=None, attach=True):
        handler = logging.FileHandler(fn)
        return LoggingHandlerBuilder.configHandler(
            handler, level, format, formatter, logger, attach)

    @staticmethod
    def createStreamHandler(stream, level, format, formatter=logging.Formatter, logger=None, attach=True):
        handler = logging.StreamHandler(stream)
        return LoggingHandlerBuilder.configHandler(handler, level, format, formatter, logger, attach)


class FileLogger:

    def __init__(self, output_dir: str = "/tmp/logs", name: str = "Logger", global_rank: int = 0, local_rank: int = 0, formatter=logging.Formatter):
        self.output_dir = output_dir
        self.name = name
        self._global_rank = global_rank
        self._local_rank = local_rank
        self._formatter = formatter

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.logger = self._build()

    def _create_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(logging.DEBUG)
        return self.logger

    def _addFileHandlers(self, logger):
        # Log
        LoggingHandlerBuilder.createFileHandler(
            self.output_dir + f'/info-{self._global_rank}.log', logging.INFO, LoggingFormats.MSG, self._formatter, self.logger,  attach=True)
        # Warn
        LoggingHandlerBuilder.createFileHandler(
            self.output_dir + f'/warn-{self._global_rank}.log', logging.WARN, LoggingFormats.MSG, self._formatter, self.logger, attach=True)
        # Debug
        LoggingHandlerBuilder.createFileHandler(
            self.output_dir + f'/debug-{self._global_rank}.log', logging.DEBUG, LoggingFormats.TIME, self._formatter, self.logger, attach=True)

    def _addConsoleHandler(self, logger):
        level = logging.DEBUG if self._local_rank == 0 else logging.WARN
        LoggingHandlerBuilder.createStreamHandler(
            level, LoggingFormats.MSG, self._formatter, logger, attach=True)

    def _build(self):
        # Base logger
        logger = self._create_logger()

        # Create and attach handlers for each info, warn, and debug levels
        self._addFileHandlers(logger)

        # Create and attach additional handler for console
        self._addConsoleHandler(logger)

        return logger

    def debug(self, *args_):
        self.logger.debug(*args_)

    def warn(self, *args_):
        self.logger.warn(*args_)

    def info(self, *args_):
        self.logger.info(*args_)

    def exception(self, *args_, **kwargs):
        return self.logger.exception(*args_, **kwargs)


class LogBuilder():
    def __init__(self, name='logger', level=logging.DEBUG, format=LoggingFormats.TIME, formatter=jsonlogger.JsonFormatter):
        self.name = name
        self.level = level
        self.format = format
        self.formatter = formatter
        self._logger = self._create_logger()

    def _create_logger(self):
        if not hasattr(self, 'logger'):
            logger = logging.getLogger(self.name)
            logger.setLevel(self.level)
        return logger

    def get_logger(self):
        return self._logger

    def _maybe_assign_defaults(self, **kwargs):
        vals = []
        for k, v in kwargs.items():
            if not v:
                vals.append(getattr(self, k))
            else:
                vals.append(v)
        return vals

    def addFileHandler(self, level=None, format=None, formatter=None, output_dir='./logs', filename='logs.log', logger=None):
        # Log
        logger = logger or self._logger
        level, format, formatter = self._maybe_assign_defaults(
            level=level, format=format, formatter=formatter)

        LoggingHandlerBuilder.createFileHandler(
            os.path.join(output_dir, filename), level, format, formatter, logger,  attach=True)

        return self

    def addConsoleHandler(self, stream=sys.stdout, level=None, format=None, formatter=None, logger=None):
        logger = logger or self._logger
        level, format, formatter = self._maybe_assign_defaults(
            level=level, format=format, formatter=formatter)

        LoggingHandlerBuilder.createStreamHandler(stream,
                                                  level, format, formatter, logger, attach=True)
        return self


def reader(path):
    with open(path, 'r') as f:
        yield from f


def line_to_json(line):
    return json.loads(line)


def logs_to_json(path):
    return map(line_to_json, reader(path))


def get_training_summary(path, summary_key='training_summary_json'):
    logs = list(logs_to_json(path))
    for l in reversed(logs):
        if summary_key in l:
            return pd.DataFrame(json.loads(l[summary_key]))
