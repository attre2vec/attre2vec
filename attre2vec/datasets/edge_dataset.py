"""Code for PyTorch dataset for edges."""
from __future__ import annotations

import dataclasses as dcs
import os
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pickle
from sklearn import preprocessing as sk_prep
import torch
from torch.utils import data as pt_data

from attre2vec.rw import utils as ng_rw_util


Edge = Tuple[int, int]


class CustomDataLoader(pt_data.DataLoader):
    """The built-in DataLoader can't provide multiple indexes for dataset."""

    def __init__(self, *args, **kwargs):
        """Inits CustomDataLoader."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def collate_fn(data):
        """When using multiple indexes no collate is needed."""
        return data

    @property
    def _auto_collation(self):
        """Override this for multiple indexes in single batch."""
        return False

    @property
    def _index_sampler(self):
        """Force to use always BatchSampler."""
        return self.batch_sampler


class EdgeDatasets:
    """Implementation of datasets iterator for edges."""

    def __init__(self, dataset_path: str, rws_path: str):
        """Inits EdgeDataset."""
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        with open(dataset_path, 'rb') as fin:
            self._ds = pickle.load(fin)

        self._rws_path = rws_path
        self.idx = -1

    def _read_rws(self, idx, split):
        with open(os.path.join(self._rws_path, f'{idx}.pkl'), 'rb') as fin:
            rws_all = pickle.load(fin)

        if split == 'test':
            return rws_all['test']

        return rws_all['train_val']

    def __iter__(self):
        """Returns iterator."""
        return self

    def __next__(self):
        """Returns next element from iterator."""
        self.idx += 1

        if self.idx == self.num_datasets:
            raise StopIteration

        return self

    def get_loader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
        mode: Optional[str] = None,
    ) -> pt_data.DataLoader:
        """Creates DataLoader for given split (train/val/test)."""
        return CustomDataLoader(
            dataset=self._get_split(self.idx, split, mode),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _get_split(self, idx: int, split: str, mode: str) -> EdgeData:
        """Gets dataset for given split."""
        Xy = self._ds['Xy'][idx][split]
        rws = self._read_rws(idx=idx, split=split)

        if split == 'test':
            graph = self._ds['original_graph']
            m = self._ds['M_test']
        else:
            graph = self._ds['graphs'][idx]
            m = self._ds['M'][idx]

        edge2idx = self._ds['edge2idx']
        node2idx = m['node2idx']
        rws_idx = ng_rw_util.to_idx_walks(
            rws=rws, edge2idx=edge2idx, node2idx=node2idx
        )

        H = torch.tensor(
            sk_prep.StandardScaler().fit_transform(self._ds['H'].numpy()),
            device=self._device,
        )

        return EdgeData(
            graph=graph,
            xe=torch.tensor(Xy['X'], device=self._device),
            y=torch.tensor(Xy['y'], device=self._device),
            h=H,
            edge2idx=edge2idx,
            m=m['M'],
            rws=rws,
            rws_idx=rws_idx,
        )

    @property
    def num_datasets(self):
        """Returns the number of dataset samples."""
        return self._ds['num_datasets']

    @property
    def num_classes(self):
        """Returns the number of classes in dataset."""
        return self._ds['num_cls']

    @property
    def dims(self):
        """Returns the dataset dimensions."""
        return self._ds['dims']


@dcs.dataclass
class EdgeData(pt_data.Dataset):
    """Implementation of PyTorch based dataset for edges."""

    graph: nx.Graph
    xe: torch.Tensor
    y: torch.Tensor
    h: torch.Tensor
    edge2idx: Dict[Edge, int]
    m: torch.Tensor
    rws: Dict[Edge, Dict[int, Dict[str, List[List[Edge]]]]]
    rws_idx: Dict[Edge, Dict[int, Dict[str, torch.Tensor]]]

    def __len__(self):
        """Returns the number of instances."""
        return len(self.xe)

    def __getitem__(self, index):
        """Extracts instances of given index (single/multiple)."""
        return EdgeData(
            graph=self.graph,
            xe=self.xe[index],
            y=self.y[index] if self.y is not None else None,
            h=self.h,
            edge2idx=self.edge2idx,
            m=self.m,
            rws=self.rws,
            rws_idx=self.rws_idx,
        )
