"""Script for evaluation of baseline model."""
import argparse
import importlib
import multiprocessing as mp
import os
import pickle
import yaml

import ge
from gem.embedding import sdne
import networkx as nx
import node2vec as n2v
import numpy as np
from sklearn import decomposition as sk_dc
from sklearn import pipeline as sk_pipe
from sklearn import preprocessing as sk_prep
from tqdm import tqdm

from attre2vec import utils as ae_utils


def get_args() -> argparse.Namespace:
    """Gets arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-cc',
        '--common-config',
        required=True,
        help='Path to common config file',
    )
    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Path to config file',
    )
    return parser.parse_args()


def make_clf(path, args):
    """Creates a classifier given config."""
    module, model = path.rsplit('.', maxsplit=1)
    clf_class = getattr(importlib.import_module(module), model)

    clf = clf_class(**args)
    return clf


def train_model(cfg, graph, is_pubmed):
    """Trains given model and returns node embeddings."""
    model_fns = {
        'deep_walk': lambda g, d: train_node2vec(graph=g, dim=d, p=1, q=1),
        'node2vec': lambda g, d: train_node2vec(graph=g, dim=d, p=4, q=1),
        'sdne': lambda g, d: train_sdne(graph=g, dim=d, is_pubmed=is_pubmed),
        'struc2vec': lambda g, d: train_struc2vec(graph=g, dim=d),
    }

    mtype = cfg['type']
    mdim = cfg['dim']

    if mtype not in model_fns.keys():
        raise KeyError(f'No such model: \"{mtype}\"')

    return model_fns[mtype](graph, mdim)


def train_node2vec(graph, dim, p, q):
    """Obtains node embeddings using Node2vec."""
    emb = n2v.Node2Vec(
        graph=graph,
        dimensions=dim,
        workers=mp.cpu_count(),
        p=p,
        q=q,
        quiet=True,
    ).fit()

    emb = {
        node_id: emb.wv[str(node_id)]
        for node_id in sorted(graph.nodes())
    }
    return emb


def train_sdne(graph, dim, is_pubmed):
    """Obtains node embeddings using SDNE."""
    mod = sdne.SDNE(
        d=dim,
        beta=5, alpha=1e-5,
        nu1=1e-6, nu2=1e-6,
        K=3, n_units=[50, 15],
        n_iter=200 if not is_pubmed else 50, xeta=0.01,
        n_batch=500,
    )
    mod.learn_embedding(graph=graph, is_weighted=False)
    np_emb = mod.get_embedding()

    emb = {
        node_id: np_emb[node_id]
        for node_id in sorted(graph.nodes())
    }
    return emb


def train_struc2vec(graph, dim):
    """Obtains node embedding using Struc2Vec."""
    g = nx.relabel_nodes(G=graph, mapping={n: str(n) for n in graph.nodes()})
    model = ge.Struc2Vec(g, workers=1, verbose=0)
    model.train(embed_size=dim, window_size=5, iter=5)
    e = model.get_embeddings()

    emb = {
        node_id: e[str(node_id)]
        for node_id in sorted(graph.nodes())
    }
    return emb


def to_edge_emb(op_name, edges, node_embs):
    """Converts node embeddings to edge embeddings."""
    e_emb = {}

    for u, v in edges:
        if op_name == 'avg':
            e_emb[(u, v)] = np.mean([node_embs[u], node_embs[v]], axis=0)
        elif op_name == 'hadamard':
            e_emb[(u, v)] = node_embs[u] * node_embs[v]
        elif op_name == 'l1':
            e_emb[(u, v)] = np.abs(node_embs[u] - node_embs[v])
        elif op_name == 'l2':
            e_emb[(u, v)] = np.power(node_embs[u] - node_embs[v], 2)

    return e_emb


def prepare_x_y(dim_red, concat, ds, idx, edge_embs=None):
    """Prepares input features and target classes for given scenario."""
    H = sk_prep.StandardScaler().fit_transform(ds['H'].numpy())
    edge_fts = {
        tt: np.array([
            H[ds['edge2idx'][e]]
            for e in ds['Xy'][idx][tt]['X']
        ])
        for tt in ('train', 'val', 'test')
    }

    if edge_embs is None:  # Simple features
        X = edge_fts
    else:  # Node embedding models (DeepWalk, Node2vec, ...)
        edge_embs = {
            tt: np.array([edge_embs[e] for e in ds['Xy'][idx][tt]['X']])
            for tt in ('train', 'val', 'test')
        }

        if concat:  # Use embedding models with initial edge features
            X = {
                tt: np.concatenate((edge_embs[tt], edge_fts[tt]), axis=1)
                for tt in ('train', 'val', 'test')
            }
        else:  # Use only embedding models
            X = edge_embs

    X = apply_dim_reduction(X, dim=dim_red)
    y = {tt: ds['Xy'][idx][tt]['y'] for tt in ('train', 'val', 'test')}

    return X, y


def apply_dim_reduction(X, dim):
    """Applied dimensionality reduction if needed."""
    if dim == -1:
        return X

    return {
        k: sk_pipe.Pipeline(steps=[
            ('scaler', sk_prep.StandardScaler()),
            ('pca', sk_dc.PCA(n_components=dim)),
        ]).fit_transform(v)
        for k, v in X.items()
    }


def main():
    """Runs the script."""
    args = get_args()

    with open(args.common_config, 'r') as fin:
        common_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    with open(args.config, 'r') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_path = common_cfg['paths']['input']['dataset']
    vectors_path = common_cfg['paths']['output']['vectors']
    metrics_path = common_cfg['paths']['output']['metrics']

    clf_cfg = common_cfg['base_clf']

    with open(dataset_path, 'rb') as fin:
        ds = pickle.load(fin)

    G = ds['original_graph']

    if cfg['model']['type'] in ('simple', 'graphsage'):
        edge_embs = None
    else:
        node_embs = train_model(
            cfg=cfg['model'],
            graph=G,
            is_pubmed='pubmed' in dataset_path
        )
        edge_embs = to_edge_emb(
            op_name=cfg['model']['op'],
            edges=G.edges(),
            node_embs=node_embs,
        )

    for scenario in tqdm(cfg['scenarios'], desc='Scenarios'):
        metrics = []

        for ds_idx in tqdm(range(ds['num_datasets']),
                           desc='Datasets',
                           leave=False):
            if cfg['model']['type'] == 'graphsage':
                ds_name = (
                    dataset_path
                    .replace('data/datasets/', '')
                    .replace('.pkl', '')
                )
                edge_embs = to_edge_emb(
                    op_name=cfg['model']['op'],
                    edges=G.edges(),
                    node_embs=np.load(f'data/graphsage/{ds_name}-{ds_idx}.npy')
                )

            X, y = prepare_x_y(
                dim_red=scenario['dim-reduction'],
                concat=scenario.get('concat', False),
                ds=ds,
                idx=ds_idx,
                edge_embs=edge_embs
            )

            # Train
            model = make_clf(path=clf_cfg['module'], args=clf_cfg['args'])
            model.fit(X['train'], y['train'])

            # Eval
            ds_metrics = {}
            for tt in ('train', 'val', 'test'):
                ds_metrics[tt] = ae_utils.calc_metrics(
                    y_score=model.predict_proba(X[tt]),
                    y_pred=model.predict(X[tt]),
                    y_true=y[tt],
                    max_cls=ds['num_cls'],
                )

            metrics.append(ds_metrics)

            # Save vectors
            vecs = [
                *X['train'].tolist(),
                *X['val'].tolist(),
                *X['test'].tolist(),
            ]
            vp = (
                vectors_path
                .replace('${NAME}', 'BL_' + scenario['name'])
                .replace('${IDX}', str(ds_idx))
            )
            os.makedirs(os.path.dirname(vp), exist_ok=True)

            with open(vp, 'wb') as fout:
                pickle.dump(vecs, fout)

        # Save
        mtrp = metrics_path.replace('${NAME}', scenario['name'])
        os.makedirs(os.path.dirname(mtrp), exist_ok=True)
        with open(mtrp, 'wb') as fout:
            pickle.dump(metrics, fout)


if __name__ == '__main__':
    main()
