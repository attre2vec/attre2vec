"""Script for running hyperparameter search."""
import argparse
import json
import os
import pickle
from time import time

import networkx as nx
import pandas as pd
import pytorch_lightning as ptl
from pytorch_lightning import callbacks as ptl_cb
from pytorch_lightning import loggers as ptl_log
import torch
from tqdm.auto import tqdm

from attre2vec import callbacks as ae_cb
from attre2vec.datasets import edge_dataset as eds
from attre2vec import model as ae_mod
from attre2vec.rw import walkers as ae_walkers
from attre2vec.rw import utils as ae_rw_util


def to_undirected(g: nx.DiGraph) -> nx.DiGraph:
    """Creates directed graph with twice many edges (to simulate undirected)."""
    return nx.DiGraph(g.to_undirected())


def make_random_walks(dataset_name, k, L):
    """Computes random walks."""
    with open(f'data/datasets/{dataset_name}.pkl', 'rb') as fin:
        dataset = pickle.load(fin)

    cfg = {
        'walker-name': 'uniform',
        'args': {'walk_length': L, 'num_walks': k}
    }

    output_path = f'data/hps/rw/{dataset_name}_k_{k}_L_{L}/'

    if os.path.exists(output_path):
        print("Done")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    times = []

    for idx in tqdm(range(dataset['num_datasets']), desc='Dataset', leave=False):  # noqa
        orig_graph = to_undirected(dataset['original_graph'])
        split_graph = to_undirected(dataset['graphs'][idx])

        walker = ae_walkers.make_walker(cfg)

        start_time = time()

        rws = {
            tt: ae_rw_util.precompute_rws_par(
                g=graph,
                xe=list(graph.edges()),
                walker=walker,
                num_workers=8,
            )
            for tt, graph in (('train_val', split_graph), ('test', orig_graph))
        }

        end_time = time()

        times.append(end_time - start_time)

        with open(os.path.join(output_path, f'{idx}.pkl'), 'wb') as fout:
            pickle.dump(rws, fout)

        del rws

    return times


def run_model(dataset_name, k, L, model_name, emb_dim, mixing):
    """Trains and evalutes the model."""
    aggregators = {
        'Avg': 'SimpleAverageAggregator',
        'GRU': 'GRUAggregator',
    }

    cfg = {
        'dataset_path': f'data/datasets/{dataset_name}.pkl',
        'rws_path': f'data/hps/rw/{dataset_name}_k_{k}_L_{L}/',

        'model_path': f'data/hps/model/{dataset_name}/{model_name}_k_{k}_L_{L}_d_{emb_dim}_mixing_{mixing}/',  # noqa
        'logs_path': f'data/hps/logs/{dataset_name}/{model_name}_k_{k}_L_{L}_d_{emb_dim}_mixing_{mixing}/',  # noqa

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
                'epochs': 150,
                'early_stopping': 15,
                'learning_rate': 1e-3,
                'weight_decay': 0,
                'batch_size': 32,
                'mixing': mixing,
            },
        },
    }

    metrics_path = f'data/hps/results-{dataset_name}-{model_name}.csv'

    # Datasets samples
    datasets = eds.EdgeDatasets(
        dataset_path=cfg['dataset_path'], rws_path=cfg['rws_path']
    )

    # Hyperparameters
    emb_dim = cfg['model']['args'].pop('emb-dim')
    hparams = argparse.Namespace(
        **cfg['model']['args'],
        **cfg['model']['training'],
        **{
            f'dims_{k}': v for k, v in datasets.dims.items()
        },
        dims_emb=emb_dim,
        num_cls=datasets.num_classes,
        base_clf_cfg=json.dumps(cfg['base_clf']),
        **{
            f'samples_{k}': v for k, v in cfg['samples'].items()
        },
    )

    metrics_out = open(metrics_path, 'a')

    for dataset in tqdm(datasets, desc='Datasets', total=datasets.num_datasets):
        # Model path
        mp = os.path.join(cfg['model_path'], '%d.pkl' % dataset.idx)
        os.makedirs(os.path.dirname(mp), exist_ok=True)

        # Logs path
        lp = cfg['logs_path']
        os.makedirs(lp, exist_ok=True)

        # Callbacks
        logger = ptl_log.TensorBoardLogger(
            save_dir=lp,
            name='',
            version=str(dataset.idx),
        )

        es = ptl_cb.EarlyStopping(
            monitor='auc',
            patience=hparams.early_stopping * 2,
            mode='max',
            verbose=False,
            strict=True,
        )

        chpkt = ae_cb.CustomModelCheckpoint(
            filepath=mp,
            monitor='auc',
            save_top_k=1,
            mode='max',
            period=1,
            verbose=False,
        )

        # Build model
        ae_model = ae_mod.UnsupervisedModel(hparams=hparams)

        # Create loaders
        tr_dl = dataset.get_loader(
            split='train',
            batch_size=hparams.batch_size,
            shuffle=True,
        )
        val_dl = dataset.get_loader(
            split='val',
            batch_size=hparams.batch_size,
            shuffle=False,
        )

        # Train
        trainer = ptl.Trainer(
            logger=logger,
            checkpoint_callback=chpkt,
            early_stop_callback=es,
            max_epochs=cfg['model']['training']['epochs'],
            progress_bar_refresh_rate=1,
            num_sanity_val_steps=0,
            gpus=1 if torch.cuda.is_available() else None,
            weights_summary=None,
        )
        trainer.fit(
            model=ae_model,
            train_dataloader=tr_dl,
            val_dataloaders=val_dl,
        )

        # Evaluate
        tr_dl = dataset.get_loader(  # Redefine train dl to be non-shuffled!
            split='train',
            batch_size=hparams.batch_size,
            shuffle=False,
        )
        test_dl = dataset.get_loader(
            split='test',
            batch_size=hparams.batch_size,
            shuffle=False,
        )
        best_ae_model = ae_mod.UnsupervisedModel.load_from_checkpoint(
            checkpoint_path=list(chpkt.best_k_models.keys())[0]
        )
        best_ae_model.eval()
        best_ae_model.freeze()

        for name, dl in (('train', tr_dl), ('val', val_dl), ('test', test_dl)):
            m = best_ae_model.eval_on_base_clf(data=list(dl))
            acc = m['accuracy']
            auc = m['auc']
            f1 = m['macro avg']['f1-score']

            metrics_out.write(
                f'{dataset_name};{k};{L};{model_name};{emb_dim};{mixing};'
                f'{dataset.idx};{name};{acc};{auc};{f1}\n'
            )

    metrics_out.close()


def main():
    """Runs the script."""
    SEARCH_SPACE = {
        'L': [4, 8, 16],
        'k': [4, 8, 16],
        'd': [16, 32, 64],
        'mixing': [0, 0.25, 0.5, 0.75, 1],
    }

    DEFAULT_PARAMETERS = {'L': 8, 'k': 16, 'd': 64, 'mixing': 0.5}

    DATASETS = ['cora', 'citeseer', 'pubmed']
    MODEL_NAMES = ['Avg', 'GRU']

    rw_times = {}

    for dataset_name in tqdm(DATASETS, desc='precompute random walks'):
        # Random walks
        for k in tqdm(SEARCH_SPACE['k'], desc='number of random walks', leave=False):  # noqa
            ts = make_random_walks(
                dataset_name=dataset_name,
                k=k,
                L=DEFAULT_PARAMETERS['L'],
            )

            if ts:
                rw_times[(dataset_name, k, DEFAULT_PARAMETERS['L'])] = ts

        for L in tqdm(SEARCH_SPACE['L'], desc='random walk length', leave=False):  # noqa
            ts = make_random_walks(
                dataset_name=dataset_name,
                k=DEFAULT_PARAMETERS['k'],
                L=L,
            )
            if ts:
                rw_times[(dataset_name, DEFAULT_PARAMETERS['k'], L)] = ts

    pd.DataFrame.from_records(
        data=[(*k, v) for k, v in rw_times.items()],
        columns=['dataset', 'k', 'L', 'times']
    ).to_csv('data/hps/rw-times.csv')

    for model_name in tqdm(MODEL_NAMES, desc='train model / model'):
        for dataset_name in tqdm(DATASETS, desc='train model / dataset'):
            # Train model
            for k in tqdm(SEARCH_SPACE['k'], desc='number of random walks', leave=False):  # noqa
                run_model(
                    dataset_name=dataset_name,
                    k=k,
                    L=DEFAULT_PARAMETERS['L'],
                    model_name=model_name,
                    emb_dim=DEFAULT_PARAMETERS['d'],
                    mixing=DEFAULT_PARAMETERS['mixing'],
                )

            for L in tqdm(SEARCH_SPACE['L'], desc='random walk length', leave=False):  # noqa
                if L == DEFAULT_PARAMETERS['L']:  # Already computed
                    continue

                run_model(
                    dataset_name=dataset_name,
                    k=DEFAULT_PARAMETERS['k'],
                    L=L,
                    model_name=model_name,
                    emb_dim=DEFAULT_PARAMETERS['d'],
                    mixing=DEFAULT_PARAMETERS['mixing'],
                )

            for d in tqdm(SEARCH_SPACE['d'], desc='embedding dimension', leave=False):  # noqa
                if d == DEFAULT_PARAMETERS['d']:
                    continue

                run_model(
                    dataset_name=dataset_name,
                    k=DEFAULT_PARAMETERS['k'],
                    L=DEFAULT_PARAMETERS['L'],
                    model_name=model_name,
                    emb_dim=d,
                    mixing=DEFAULT_PARAMETERS['mixing'],
                )

            for mixing in tqdm(SEARCH_SPACE['mixing'], desc='loss function mixing', leave=False):  # noqa
                if mixing == DEFAULT_PARAMETERS['mixing']:
                    continue

                run_model(
                    dataset_name=dataset_name,
                    k=DEFAULT_PARAMETERS['k'],
                    L=DEFAULT_PARAMETERS['L'],
                    model_name=model_name,
                    emb_dim=DEFAULT_PARAMETERS['d'],
                    mixing=mixing,
                )


if __name__ == '__main__':
    main()
