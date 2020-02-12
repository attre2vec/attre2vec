"""Utils for paper graph."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import pickle

import networkx as nx
import pytorch_lightning as ptl
import torch
from torch.utils import data as pt_data

from attre2vec.datasets import edge_dataset as eds
from attre2vec import model as ng_mod
from attre2vec.rw import walkers as ng_walkers
from attre2vec.rw import utils as ng_rw_util


def to_undirected(g: nx.DiGraph) -> nx.DiGraph:
    """Creates directed graph with twice many edges (to simulate undirected)."""
    return nx.DiGraph(g.to_undirected())


def make_random_walks(dataset, k, L):
    """Computes random walks."""
    cfg = {
        'walker-name': 'uniform',
        'args': {'walk_length': L, 'num_walks': k}
    }

    graph = to_undirected(dataset['graph'])

    walker = ng_walkers.make_walker(cfg)

    rws = ng_rw_util.precompute_rws_par(
        g=graph,
        xe=list(graph.edges()),
        walker=walker,
        num_workers=1,
    )
    return rws


class TransductiveEdgeDataset:
    """Implementation of transductive edge dataset."""

    def __init__(self, dataset_path: str, rws_path: str):
        """Inits EdgeDataset."""
        with open(dataset_path, 'rb') as fin:
            self._ds = pickle.load(fin)

        with open(rws_path, 'rb') as fin:
            self._rws = pickle.load(fin)

    def get_loader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool,
    ) -> pt_data.DataLoader:
        """Creates DataLoader."""
        graph = self._ds['graph']
        edge2idx = self._ds['edge2idx']
        rws_idx = ng_rw_util.to_idx_walks(
            rws=self._rws, edge2idx=edge2idx, node2idx=defaultdict(lambda: 0)
        )

        M = torch.zeros((1, self._ds['dims']['node']), dtype=torch.float)

        dataset = eds.EdgeData(
            graph=graph,
            xe=torch.tensor(self._ds['xe']),
            y=torch.tensor(self._ds['y']),
            h=torch.tensor(self._ds['H'], dtype=torch.float),
            edge2idx=edge2idx,
            m=M,
            rws=self._rws,
            rws_idx=rws_idx,
        )

        return eds.CustomDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @property
    def num_classes(self):
        """Returns the number of classes in dataset."""
        return self._ds['num_cls']

    @property
    def dims(self):
        """Returns the dataset dimensions."""
        return self._ds['dims']


def run_model(dataset_name, model_name, emb_dim, mixing, epochs):
    """Trains and evalutes the model."""
    aggregators = {
        'Avg': 'SimpleAverageAggregator',
        'GRU': 'GRUAggregator',
    }

    cfg = {
        'dataset_path': f'../../data/ablation/{dataset_name}.pkl',
        'rws_path': f'../../data/ablation/rws.pkl',

        'model_path': f'../../data/ablation/model/{model_name}_d_{emb_dim}_mixing_{mixing}.pkl',  # noqa: E501

        'base_clf': {
            'path': 'sklearn.linear_model.LogisticRegression',
            'args': {
                'max_iter': 500,
                'multi_class': 'multinomial',
                'n_jobs': -1,
            },
        },

        'samples': {'pos': 5, 'neg': 10},

        'model': {
            'args': {
                'aggregator': aggregators[model_name],
                'encoder': 'Encoder',
                'decoder': 'MLPDecoder',
                'emb-dim': emb_dim,
            },
            'training': {
                'epochs': epochs,
                'early_stopping': None,
                'learning_rate': 1e-3,
                'weight_decay': 0,
                'batch_size': 2,
                'mixing': mixing,
            },
        },
    }

    # Datasets samples
    tdataset = TransductiveEdgeDataset(
        dataset_path=cfg['dataset_path'], rws_path=cfg['rws_path']
    )

    # Hyperparameters
    emb_dim = cfg['model']['args'].pop('emb-dim')
    hparams = argparse.Namespace(
        **cfg['model']['args'],
        **cfg['model']['training'],
        **{
            f'dims_{k}': v for k, v in tdataset.dims.items()
        },
        dims_emb=emb_dim,
        num_cls=tdataset.num_classes,
        base_clf_cfg=json.dumps(cfg['base_clf']),
        **{
            f'samples_{k}': v for k, v in cfg['samples'].items()
        },
    )

    # Model path
    os.makedirs(os.path.dirname(cfg['model_path']), exist_ok=True)

    # Build model
    ng_model = ng_mod.UnsupervisedModel(hparams=hparams)
    ng_model.validation_step = None

    # Create loaders
    tr_dl = tdataset.get_loader(
        split='all',  # Doesn't matter
        batch_size=hparams.batch_size,
        shuffle=True,
    )
    tr_dl_sorted = tdataset.get_loader(
        split='all',
        batch_size=hparams.batch_size,
        shuffle=False,
    )

    ng_model.set_base_clf_data(train_data=tr_dl_sorted, val_data=None)

    # Train
    trainer = ptl.Trainer(
        early_stop_callback=None,
        val_percent_check=0,
        fast_dev_run=False,
        max_epochs=epochs,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        weights_summary=None,
    )
    trainer.fit(model=ng_model, train_dataloader=tr_dl)

    # Evaluate
    ng_model.eval()
    ng_model.freeze()

    ng_model.set_base_clf_data(train_data=tr_dl_sorted, val_data=None)
    ng_model.train_base_clf()

    m = ng_model.eval_on_base_clf(data=list(tr_dl_sorted))

    acc = m['accuracy']
    auc = m['auc']
    f1 = m['macro avg']['f1-score']

    print('Acc:', acc)
    print('AUC:', auc)
    print('F1:', f1)
    print('CM:', m['cm'])

    embs, labels = ng_model.data_to_emb_labels(data=list(tr_dl_sorted))
    return embs, auc
