import numpy as np
from utils.custom_typing import *
class Metric():
    '''
    Computes the running average of a scalar quantity over a set of batches (single evaluation of the metric).
    A metric has 2 values, one for training and one for validation.
    A metric is associated to a name (for logging and readability) and an optional tensorboard logger that will output the values once the metric is final (i.e an epoch is finished)
    '''
    def __init__(self, metric_name, tensorboard):
        self.running_value_train = 0.0
        self.running_value_val = 0.0
        self.seen_batches_train = 0
        self.seen_batches_val = 0
        self.tb = tensorboard
        self.metric_name = metric_name

    def reset(self) -> None:
        self.running_value_train = 0.0
        self.running_value_val = 0.0
        self.seen_batches_train = 0
        self.seen_batches_val = 0

    def new_value_train(self, value: float) -> None:
        self.running_value_train += value
        self.seen_batches_train += 1

    def new_value_val(self, value: float) -> None:
        self.running_value_val += value
        self.seen_batches_val += 1

    def get_value_train(self) -> float:
        return self.running_value_train / self.seen_batches_train

    def get_value_val(self) -> float:
        return self.running_value_val / self.seen_batches_val

    def get_train_value_as_display(self, roundoff: UnsignedInt = 6) -> float:
        return np.round(self.get_value_train(), decimals=roundoff)

    def write_to_tensorboard(self, epoch: UnsignedInt) -> None:
        if self.tb is not None:
            self.tb.add_scalar('{}/train'.format(self.metric_name), self.get_value_train(), epoch)
            self.tb.add_scalar('{}/val'.format(self.metric_name), self.get_value_val(), epoch)

class MetricsList():
    '''
    A set of metrics: follows the same interface as a metric, but expects a list of metrics as input for creation
    '''
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    def new_values_train(self, values: List[float]) -> None:
        for m, v in zip(self.metrics, values):
            m.new_value_train(v)

    def new_values_val(self, values: List[float]) -> None:
        for m, v in zip(self.metrics, values):
            m.new_value_val(v)

    def get_values_train(self) -> List[float]:
        return [m.get_value_train() for m in self.metrics]

    def get_values_val(self) -> List[float]:
        return [m.get_value_val() for m in self.metrics]

    def get_train_values_as_display(self, roundoff=6) -> List[float]:
        return [m.get_train_value_as_display(roundoff) for m in self.metrics]

    def write_to_tensorboard(self, epoch) -> None:
        for m in self.metrics:
            m.write_to_tensorboard(epoch)

    def values_as_dict(self) -> Dict[str, float]:
        d = {}
        for m in self.metrics:
            d[m.metric_name] = m.get_value_train()
            d['val_{}'.format(m.metric_name)] = m.get_value_val()
        return d
