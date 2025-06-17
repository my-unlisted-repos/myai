from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..data import DS

DATASETS_ROOT = '/var/mnt/hdd/Datasets'

class Dataset(DS, ABC):
    def submission(self, fname, model, *args, **kwargs):
        """makes a submission file, if applicable"""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `make_val_submission`")

    def preprocess(self, inputs: Sequence, *args, **kwargs):
        """loads and prepocesses a sequence of inputs, applies same transforms as :code:`load`."""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement `preprocess`")

    def inference(self, model, inputs: Sequence, *args, **kwargs):
        """run inference on a sequence of input, applies same transforms as :code:`load`."""
        return model(self.preprocess(inputs, *args, **kwargs))