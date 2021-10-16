from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricMonitor(Callback):
    def __init__(self, stage='train', metric=None, logger=None, logging_interval=None, log_momentum: bool = False):

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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if self.stage == 'train' or self.stage == 'both':
            if self.logging_interval != "epoch":
                self.logger.report_scalar(title=f"train_{self.metric}_by_step", series='train', value=outputs[self.metric],
                                          iteration=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stage == 'train' or self.stage == 'both':
            if self.logging_interval != "step":
                outputs = pl_module.train_epoch_outputs
                self.logger.report_scalar(title=f"train_{self.metric}_by_epoch", series='train', value=outputs[self.metric],
                                          iteration=trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if self.stage == 'valid' or self.stage == 'both':
            if self.logging_interval != "epoch":
                self.logger.report_scalar(title=f"valid_{self.metric}_by_step", series='valid', value=outputs[self.metric],
                                          iteration=trainer.global_step)


    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stage == 'valid' or self.stage == 'both':
            if self.logging_interval != "step":
                outputs = pl_module.valid_epoch_outputs
                print(outputs)
                self.logger.report_scalar(title=f"valid_{self.metric}_by_epoch", series='valid', value=outputs[self.metric],
                                          iteration=trainer.current_epoch)


class LogAUCMonitor(MetricMonitor):
    def __init__(self, stage = 'train', logger=None, logging_interval=None, log_momentum: bool = False):
        super(LogAUCMonitor, self).__init__(stage = stage, metric="logAUC", logger=logger, logging_interval=logging_interval, log_momentum=log_momentum)


class LossMonitor(MetricMonitor):
    def __init__(self, stage = 'train', logger=None, logging_interval=None, log_momentum: bool = False):
        super(LossMonitor, self).__init__(stage = stage, metric="loss", logger=logger, logging_interval=logging_interval, log_momentum=log_momentum)
