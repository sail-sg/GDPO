import copy
import inspect
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from torch_geometric.data import (
    Data,
    Dataset,
)
from torch_geometric.loader import DataLoader
# from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes, OptTensor

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs):
        super().__init__()

        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' found on this machine. "
                "Run 'pip install pytorch_lightning' to install the library.")

        if not has_val:
            self.val_dataloader = None

        if not has_test:
            self.test_dataloader = None

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers',
                          kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            warnings.warn(f"The 'shuffle={kwargs['shuffle']}' option is "
                          f"ignored in '{self.__class__.__name__}'. Remove it "
                          f"from the argument list to disable this warning")
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kwargs_repr(**self.kwargs)})'


# class LightningData(LightningDataModule):
#     def __init__(
#         self,
#         data: Union[Data, HeteroData],
#         has_val: bool,
#         has_test: bool,
#         loader: str = 'neighbor',
#         graph_sampler: Optional[BaseSampler] = None,
#         eval_loader_kwargs: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         kwargs.setdefault('batch_size', 1)
#         kwargs.setdefault('num_workers', 0)

#         if graph_sampler is not None:
#             loader = 'custom'

#         # For full-batch training, we use reasonable defaults for a lot of
#         # data-loading options:
#         if loader not in ['full', 'neighbor', 'link_neighbor', 'custom']:
#             raise ValueError(f"Undefined 'loader' option (got '{loader}')")

#         if loader == 'full' and kwargs['batch_size'] != 1:
#             warnings.warn(f"Re-setting 'batch_size' to 1 in "
#                           f"'{self.__class__.__name__}' for loader='full' "
#                           f"(got '{kwargs['batch_size']}')")
#             kwargs['batch_size'] = 1

#         if loader == 'full' and kwargs['num_workers'] != 0:
#             warnings.warn(f"Re-setting 'num_workers' to 0 in "
#                           f"'{self.__class__.__name__}' for loader='full' "
#                           f"(got '{kwargs['num_workers']}')")
#             kwargs['num_workers'] = 0

#         if loader == 'full' and kwargs.get('sampler') is not None:
#             warnings.warn("'sampler' option is not supported for "
#                           "loader='full'")
#             kwargs.pop('sampler', None)

#         if loader == 'full' and kwargs.get('batch_sampler') is not None:
#             warnings.warn("'batch_sampler' option is not supported for "
#                           "loader='full'")
#             kwargs.pop('batch_sampler', None)

#         super().__init__(has_val, has_test, **kwargs)

#         if loader == 'full':
#             if kwargs.get('pin_memory', False):
#                 warnings.warn(f"Re-setting 'pin_memory' to 'False' in "
#                               f"'{self.__class__.__name__}' for loader='full' "
#                               f"(got 'True')")
#             self.kwargs['pin_memory'] = False

#         self.data = data
#         self.loader = loader

#         # Determine sampler and loader arguments ##############################

#         if loader in ['neighbor', 'link_neighbor']:

#             # Define a new `NeighborSampler` to be re-used across data loaders:
#             sampler_kwargs, self.loader_kwargs = split_kwargs(
#                 self.kwargs,
#                 NeighborSampler,
#             )
#             sampler_kwargs.setdefault('share_memory',
#                                       self.kwargs['num_workers'] > 0)
#             self.graph_sampler = NeighborSampler(data, **sampler_kwargs)

#         elif graph_sampler is not None:
#             sampler_kwargs, self.loader_kwargs = split_kwargs(
#                 self.kwargs,
#                 graph_sampler.__class__,
#             )
#             if len(sampler_kwargs) > 0:
#                 warnings.warn(f"Ignoring the arguments "
#                               f"{list(sampler_kwargs.keys())} in "
#                               f"'{self.__class__.__name__}' since a custom "
#                               f"'graph_sampler' was passed")
#             self.graph_sampler = graph_sampler

#         else:
#             assert loader == 'full'
#             self.loader_kwargs = self.kwargs

#         # Determine validation sampler and loader arguments ###################

#         self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
#         if eval_loader_kwargs is not None:
#             # If the user wants to override certain values during evaluation,
#             # we shallow-copy the graph sampler and update its attributes.
#             if hasattr(self, 'graph_sampler'):
#                 self.eval_graph_sampler = copy.copy(self.graph_sampler)

#                 eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
#                     eval_loader_kwargs,
#                     self.graph_sampler.__class__,
#                 )
#                 for key, value in eval_sampler_kwargs.items():
#                     setattr(self.eval_graph_sampler, key, value)

#             self.eval_loader_kwargs.update(eval_loader_kwargs)

#         elif hasattr(self, 'graph_sampler'):
#             self.eval_graph_sampler = self.graph_sampler

#         self.eval_loader_kwargs.pop('sampler', None)
#         self.eval_loader_kwargs.pop('batch_sampler', None)

#     @property
#     def train_shuffle(self) -> bool:
#         shuffle = self.loader_kwargs.get('sampler', None) is None
#         shuffle &= self.loader_kwargs.get('batch_sampler', None) is None
#         return shuffle

#     def prepare_data(self):
#         if self.loader == 'full':
#             try:
#                 num_devices = self.trainer.num_devices
#             except AttributeError:
#                 # PyTorch Lightning < 1.6 backward compatibility:
#                 num_devices = self.trainer.num_processes
#                 num_devices = max(num_devices, self.trainer.num_gpus)

#             if num_devices > 1:
#                 raise ValueError(
#                     f"'{self.__class__.__name__}' with loader='full' requires "
#                     f"training on a single device")
#         super().prepare_data()

#     def full_dataloader(self, **kwargs) -> DataLoader:
#         warnings.filterwarnings('ignore', '.*does not have many workers.*')
#         warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

#         return torch.utils.data.DataLoader(
#             [self.data],
#             collate_fn=lambda xs: xs[0],
#             **kwargs,
#         )

#     def __repr__(self) -> str:
#         kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
#         return f'{self.__class__.__name__}({kwargs})'


class LightningDataset(LightningDataModule):
    r"""Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
    :class:`pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a :obj:`datamodule` for multi-GPU graph-level
    training via :lightning:`null`
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.DataLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPStrategy` training
        strategies of :lightning:`null` `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: :obj:`None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: :obj:`None`)
        pred_dataset (Dataset, optional): The prediction dataset.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.DataLoader`.
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pred_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        super().__init__(
            has_val=val_dataset is not None,
            has_test=test_dataset is not None,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

    def dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get('sampler', None) is None
        shuffle &= self.kwargs.get('batch_sampler', None) is None

        return self.dataloader(self.train_dataset, shuffle=shuffle,
                               **self.kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.pred_dataset, shuffle=False, **kwargs)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(train_dataset=self.train_dataset,
                             val_dataset=self.val_dataset,
                             test_dataset=self.test_dataset,
                             pred_dataset=self.pred_dataset, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'