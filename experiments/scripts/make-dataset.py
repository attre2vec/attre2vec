"""Script for creation of pre-processed Cora/Citeseer dataset."""
import argparse
import os
import pickle
import yaml

from attre2vec import datasets as ae_ds


def get_args() -> argparse.Namespace:
    """Gets arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c',
        '--config',
        required=True,
        help='Path to config file',
    )
    parser.add_argument(
        '-n',
        '--name',
        required=True,
        help='Name of dataset to read',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Path to output dataset pickle',
    )
    return parser.parse_args()


def main():
    """Runs the script."""
    args = get_args()

    with open(args.config, 'r') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    ds = ae_ds.read_cora_citeseer_pubmed(
        path=f'data/raw/{args.name}',
        node_dim=cfg['node-emb-dim'],
        doc2vec_kwargs=cfg['doc2vec'],
        split_sizes=cfg['split-sizes'],
        num_datasets=cfg['num-datasets'],
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'wb') as fout:
        pickle.dump(ds, fout)


if __name__ == '__main__':
    main()
