from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricMonitor(Callback):
    def __init__(self, stage='train', metric=None, logger=None, logging_interval=None, title = None):

        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException("MetricMonitor: logging_interval should be `step` or `epoch` or `None`.")
        if metric is None:
            raise MisconfigurationException("MetricMonitor: metric is not specified")
        if stage not in ('both', 'train', 'valid'):
            raise MisconfigurationException(f"MetricMonitor: input 'stage' argument = {stage}, which cannot be recognized")
        self.logger = logger
        self.metric = metric
        self.logging_interval = logging_interval
        self.stage = stage
        self.title = title

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if self.stage == 'train' or self.stage == 'both':
            if self.logging_interval != "epoch":
                if self.title is not None:
                    title = self.title + f"train_{self.metric}_by_step"
                else:
                    title = f"train_{self.metric}_by_step"
                self.logger.report_scalar(title= title, series='train', value=outputs[self.metric],
                                          iteration=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stage == 'train' or self.stage == 'both':
            if self.logging_interval != "step":
                outputs = pl_module.train_epoch_outputs
                if self.title is not None:
                    title = self.title + f"train_{self.metric}_by_epoch"
                else:
                    title = f"train_{self.metric}_by_epoch"
                self.logger.report_scalar(title=title, series='train', value=outputs[self.metric],
                                          iteration=trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if self.stage == 'valid' or self.stage == 'both':
            if self.logging_interval != "epoch":
                if self.title is not None:
                    title = self.title + f"valid_{self.metric}_by_step"
                else:
                    title = f"valid_{self.metric}_by_step"

                self.logger.report_scalar(title=title, series='valid', value=outputs[self.metric],
                                          iteration=trainer.global_step)


    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stage == 'valid' or self.stage == 'both':
            if self.logging_interval != "step":
                outputs = pl_module.valid_epoch_outputs
                print(outputs)
                if self.title is not None:
                    title = self.title + f"valid_{self.metric}_by_epoch"
                else:
                    title = f"valid_{self.metric}_by_epoch"
                self.logger.report_scalar(title=title, series='valid', value=outputs[self.metric],
                                          iteration=trainer.current_epoch)


class LogAUCMonitor(MetricMonitor):
    def __init__(self, stage = 'valid', logger=None, logging_interval=None, title = None):
        super(LogAUCMonitor, self).__init__(stage=stage, metric="logAUC", logger=logger, logging_interval=logging_interval, title = title)


class LossMonitor(MetricMonitor):
    def __init__(self, stage = 'train', logger=None, logging_interval=None, title = None):
        super(LossMonitor, self).__init__(stage=stage, metric="loss", logger=logger, logging_interval=logging_interval, title = title)


class PPVMonitor(MetricMonitor):
    def __init__(self, stage = 'valid', logger=None, logging_interval=None, title = None):
        super(PPVMonitor, self).__init__(stage=stage, metric="ppv", logger=logger, logging_interval=logging_interval, title = title)