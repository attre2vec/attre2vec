"""Script for evaluation of Line2vec embeddings."""
import argparse
import importlib
import os
import pickle
import yaml

import numpy as np
from tqdm import tqdm

from attre2vec import utils as ae_utils


def get_args() -> argparse.Namespace:
    """Gets arguments."""
    parser = argparse.ArgumentParser()

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


def prepare_x_y(ds, idx, edge_embs):
    """Prepares input features and target classes for given scenario."""
    X = {
        tt: np.array([
            edge_embs.get(e) or edge_embs[(e[1], e[0])]
            for e in ds['Xy'][idx][tt]['X']
        ])
        for tt in ('train', 'val', 'test')
    }

    y = {tt: ds['Xy'][idx][tt]['y'] for tt in ('train', 'val', 'test')}

    return X, y


def main():
    """Runs the script."""
    args = get_args()

    with open(args.config, 'r') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_path = cfg['paths']['input']['dataset']
    embeddings_path = cfg['paths']['input']['emb']

    vectors_path = cfg['paths']['output']['vectors']
    metrics_path = cfg['paths']['output']['metrics']

    clf_cfg = cfg['base_clf']

    with open(dataset_path, 'rb') as fin:
        ds = pickle.load(fin)

    with open(embeddings_path, 'rb') as fin:
        edge_embs = pickle.load(fin)

    metrics = []

    for ds_idx in tqdm(range(ds['num_datasets']), desc='Datasets', leave=False):
        X, y = prepare_x_y(ds=ds, idx=ds_idx, edge_embs=edge_embs)

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
        vp = vectors_path.replace('${IDX}', str(ds_idx))
        os.makedirs(os.path.dirname(vp), exist_ok=True)

        with open(vp, 'wb') as fout:
            pickle.dump(vecs, fout)

    # Save
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'wb') as fout:
        pickle.dump(metrics, fout)


if __name__ == '__main__':
    main()
