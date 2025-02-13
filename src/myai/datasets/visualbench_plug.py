from collections import abc
from typing import Any, Literal
import torch

from visualbench.benchmark import Benchmark, sig
from ..data.ds import DS

class MyaiDataset(Benchmark):
    def __init__(
        self,
        dataset: Any,
        model: torch.nn.Module | sig,
        loss: abc.Callable | sig,
        batch_size: int | None,
        test_batch_size: int | None | Literal['same'] = 'same',
        train_split: float | int | None = 0.8,
        split_shuffle: bool = False,
        num_samples: int | float | None = None,
        per_class: bool = False,
        train_batch_tfms = None,
        test_batch_tfms = None,
        preload = True,
        use_tensor_dataloader = True,
        memory_efficient = True,
        stack_dtype = None,
        stack_device = None,
        save_edge_params = False,
        device = 'cuda',
        seed: int | None = 0,
    ):
        """dataset to benchmark

        Args:
            dataset (Any): any dataset module or class from myai.datasets.
            model (torch.nn.Module | sig): model.
            batch_size (int | None): batch size.
            loss (Callable | sig): loss function.
            test_batch_size (int | None | Literal[&#39;same&#39;], optional): test batch size. Defaults to 'same'.
            train_split (float | int | None, optional): test split. Defaults to 0.8.
            split_shuffle (bool, optional): whether to shuffle before test splitting. Defaults to False.
            num_samples (int | float | None, optional): subsample to this many samples. Defaults to None.
            per_class (bool, optional): whether to treat `num_samples` as per class or in total. Defaults to False.
            train_batch_tfms (_type_, optional): train batch transforms. Defaults to None.
            test_batch_tfms (_type_, optional): test batch transforms. Defaults to None.
            preload (bool, optional): preloads the dataset. Defaults to True.
            use_tensor_dataloader (bool, optional): uses fast tensor dataloader. Defaults to True.
            memory_efficient (bool, optional): memory efficient option for tensor dataloader. Defaults to True.
            stack_dtype (_type_, optional): dtype or list of dtypes for full-batch or tensor dataloader stack. Defaults to None.
            stack_device (_type_, optional): device for full-batch or tensor dataloader stack. Defaults to None.
            save_edge_params (str, optional): save_edge_params. Defaults to False.
            device (str, optional): device. Defaults to 'cuda'.
            seed (int | None, optional): random seed. Defaults to 0.
        """
        ds: DS = dataset.get()

        if num_samples is not None: ds = ds.subsample(num_samples, per_class, preload=preload, seed=seed)

        if preload and not use_tensor_dataloader:
            ds.preload_()

        if train_split is not None: dstrain, dstest = ds.split(train_split, split_shuffle, seed=seed)
        else: dstrain, dstest = ds, None

        if test_batch_size == 'same': test_batch_size = batch_size

        if batch_size is None: train_data = (dstrain.stack(stack_dtype, stack_device), )
        elif use_tensor_dataloader: train_data = dstrain.tensor_dataloader(batch_size, True, memory_efficient, seed=seed, dtype=stack_dtype, device=stack_device)
        else: train_data = dstrain.dataloader(batch_size, True, seed=seed)

        if dstest is not None:
            if test_batch_size is None: test_data = (dstest.stack(stack_dtype, stack_device), )
            elif use_tensor_dataloader: test_data = dstest.tensor_dataloader(test_batch_size, False, memory_efficient, dtype=stack_dtype, device=stack_device)
            else: test_data = dstest.dataloader(test_batch_size, False)
        else: test_data = None


        super().__init__(
            train_data = train_data,
            test_data = test_data,
            train_batch_tfms = train_batch_tfms,
            test_batch_tfms = test_batch_tfms,
            save_edge_params = save_edge_params,
            device = device,
            seed = seed,
        )

        self.model = self._save_signature(model, 'model')
        self.loss_fn = self._save_signature(loss, 'loss_fn')

    def get_loss(self):
        inputs, targets = [i.to(self.device) for i in self.batch]
        preds: torch.Tensor = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        accuracy = preds.argmax(1).eq_(targets).float().mean()
        return loss, {"accuracy": accuracy}

    def reset(self, model):
        super().reset()
        self.model = self._save_signature(model, 'model')