import torch
from pathlib import Path

from utils.custom_typing import *

class Callback(object):
    """A callback object, to be used during training of a neural net

    """
    def __init__(self):
        super(Callback, self).__init__()

    def on_batch_begin(self, batch_idx: UnsignedInt) -> None:
        """ Callback to be called before a batch is processed

        Args:
            batch_idx: Index of the batch
        """
        pass

    def on_batch_end(self, batch_idx: UnsignedInt) -> None:
        """Callback to be called when a batch has been processed

        Args:
            batch_idx: Index of the batch that has been processed
        """
        pass

    def on_epoch_begin(self, epoch_idx: UnsignedInt, metrics: Dict[str, float]) -> None:
        """Function called at the start of an epoch

        Args:
            epoch_idx: the index of the epoch that started
            metrics: The value of the training metrics
        """
        pass

    def on_epoch_end(self, epoch_idx: UnsignedInt, metrics: Dict[str, float]) -> None:
        """Function called at the end of an epoch

        Args:
            epoch_idx: the index of the epoch that started
            metrics: The value of the training metrics
        """
        pass

class DispatchCallback(Callback):
    """A dispatcher callback, that forwards function calls to subcallbacks
    """
    def __init__(self, cbs):
        """

        Args:
            cbs (List[Callbacks]): The subcallbacks to handle
        """
        super(DispatchCallback, self).__init__()
        self.cbs = cbs
    def on_batch_begin(self, batch_idx):
        for cb in self.cbs:
            cb.on_batch_begin(batch_idx)
    def on_batch_end(self, batch_idx):
        for cb in self.cbs:
            cb.on_batch_end(batch_idx)
    def on_epoch_begin(self, epoch_idx, metrics):
        for cb in self.cbs:
            cb.on_epoch_begin(epoch_idx, metrics)
    def on_epoch_end(self, epoch_idx, metrics):
        for cb in self.cbs:
            cb.on_epoch_end(epoch_idx, metrics)

class SaveModelCallback(Callback):
    '''
    A callback to save the model when a better one has been found
    '''
    def __init__(self, key: str, mode: str, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, model_path: Path, optimizer_path: Path):
        """

        Args:
            key (str): Metric key that will be watched to see if results improve
            mode (str: min|max): whether a decrease (min) or an increase (max) of the metric value is considered an improvement
            model (nn.Module): The model to save
            optimizer (torch.optim.Optimizer): The optimizer used to train the model.
            model_path (pathlib.Path): Where to save the model
            optimizer_path (pathlib.Path): Where to save the optimizer
        """        
        super(SaveModelCallback, self).__init__()
        self.key = key
        self.mode = mode
        self.model = model
        self.optimizer = optimizer
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        
        assert self.mode in ['min', 'max']
        self.current_best_value = None
        self.save()

    def on_epoch_end(self, epoch_idx, metrics):
        assert self.key in metrics
        new_value = metrics[self.key]
        should_save = self.current_best_value is None or new_value < self.current_best_value and self.mode == 'min' or new_value > self.current_best_value and self.mode == 'max'
        if should_save:
            print('Saving model!')
            self.current_best_value = new_value
            self.save()

    def save(self):
        import torch
        torch.save(self.model, str(self.model_path))
        if self.optimizer is not None and self.optimizer_path is not None:
            torch.save(self.optimizer, str(self.optimizer_path))

class SaveModelCallbackMultiKeysAND(Callback):
    '''
    Callback to save a model when multiple keys have improved
    '''
    def __init__(self, keys, modes_dict, model, optimizer, model_path, optimizer_path):
        """

        Args:
            keys (List[str]): Metric keys that will be watched to see if results improve
            mode (Dict[str: in keys -> str: min|max): whether a decrease (min) or an increase (max) of the metric value is considered an improvement
            model (nn.Module): The model to save
            optimizer (torch.optim.Optimizer): The optimizer used to train the model.
            model_path (pathlib.Path): Where to save the model
            optimizer_path (pathlib.Path): Where to save the optimizer
        """   
        super(SaveModelCallbackMultiKeysAND, self).__init__()
        self.keys = keys
        self.modes_dict = modes_dict
        for k, d in self.modes_dict.items():
            assert d in ['min', 'max']
        self.model = model
        self.optimizer = optimizer
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.current_best_values = {k: None for k in self.keys}
    def on_epoch_end(self, epoch_idx, metrics):
        relevant_metrics = {k: v for k,v in metrics.items() if k in self.keys}
        should_save = all(self._is_better(k, v) for k,v in relevant_metrics.items())
        if should_save:
            self.current_best_values = relevant_metrics
            self.save()
    def _is_better(self, k, v):
        v_old = self.current_best_values[k]
        return v_old is None or v < v_old and self.modes_dict[k] == 'min' or v > v_old and self.modes_dict[k] == 'max'
    def save(self):
        
        torch.save(self.model, str(self.model_path))
        torch.save(self.optimizer, str(self.optimizer_path))