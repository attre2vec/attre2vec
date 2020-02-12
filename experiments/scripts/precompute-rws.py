"""Script for pre-computation of random walks on datasets."""
import argparse
import os
import pickle
import yaml

import networkx as nx
from tqdm import tqdm

from attre2vec.rw import walkers as ae_walkers
from attre2vec.rw import utils as ae_rw_util


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
        '-i',
        '--input',
        required=True,
        help='Path to input dataset pickle',
    )
    parser.add_argument(
        '-o',
        '--output',
        required=True,
        help='Path to output random walks pickle',
    )
    return parser.parse_args()


def to_undirected(g: nx.DiGraph) -> nx.DiGraph:
    """Creates directed graph with twice many edges (to simulate undirected)."""
    return nx.DiGraph(g.to_undirected())


def main():
    """Runs the script."""
    args = get_args()

    with open(args.config, 'r') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    with open(args.input, 'rb') as fin:
        dataset = pickle.load(fin)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    for idx in tqdm(range(dataset['num_datasets']), desc='Dataset'):
        orig_graph = to_undirected(dataset['original_graph'])
        split_graph = to_undirected(dataset['graphs'][idx])

        walker = ae_walkers.make_walker(cfg)

        rws = {
            tt: ae_rw_util.precompute_rws_par(
                g=graph,
                xe=list(graph.edges()),
                walker=walker,
                num_workers=8,
            )
            for tt, graph in (('train_val', split_graph), ('test', orig_graph))
        }

        with open(os.path.join(args.output, f'{idx}.pkl'), 'wb') as fout:
            pickle.dump(rws, fout)

        del rws


if __name__ == '__main__':
    main()
