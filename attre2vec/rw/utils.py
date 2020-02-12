"""Implementation of random walkers."""
import multiprocessing as mp
from typing import Dict, List, Tuple

import networkx as nx
import torch
from tqdm.auto import tqdm

from attre2vec.rw import walkers as ng_walkers


def to_idx_walks(rws, edge2idx, node2idx):
    """Converts random walks to indexes in edge and node feature matrices."""
    rws_idx = {}
    for edge in tqdm(rws.keys(), desc='Index edges', leave=False):
        rws_idx[edge] = {}
        for node in edge:
            rws_idx[edge][node] = {'edge': [], 'node': []}
            for walk in rws[edge][node]:
                eidx = [edge2idx[e] for e in walk]
                nidx = [node2idx[u] for u, _ in walk]

                rws_idx[edge][node]['edge'].append(eidx)
                rws_idx[edge][node]['node'].append(nidx)

            rws_idx[edge][node]['edge'] = torch.tensor(
                rws_idx[edge][node]['edge']
            )
            rws_idx[edge][node]['node'] = torch.tensor(
                rws_idx[edge][node]['node']
            )

    return rws_idx


def precompute_rws_par(
    g: nx.DiGraph,
    xe: List[Tuple[int, int]],
    walker: ng_walkers.BaseWalker,
    num_workers: int = mp.cpu_count(),
) -> Dict[int, List[int]]:
    """Pre-computes random walks in parallel manner."""
    args = [
        dict(g=g, xe=chunk, walker=walker)
        for chunk in make_chunks(values=xe, chunk_size=len(xe) // num_workers)
    ]

    if num_workers == 1:
        rws_par = [_precompute_rws_worker_fn(a) for a in args]
    else:
        with mp.Pool(processes=num_workers) as pool:
            rws_par = list(pool.imap(_precompute_rws_worker_fn, args))

    rws = {}
    for rw in rws_par:
        for k, v in rw.items():
            rws[k] = v

    return rws


def make_chunks(values, chunk_size):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(values), chunk_size):
        yield values[i:i + chunk_size]


def _precompute_rws_worker_fn(kwargs):
    return precompute_rws(**kwargs)


def precompute_rws(
    g: nx.DiGraph,
    xe: List[Tuple[int, int, int]],
    walker: ng_walkers.BaseWalker,
) -> Dict[int, List[int]]:
    """Pre-computes random walks."""
    rws = {edge: {} for edge in xe}

    for u, v in tqdm(xe, desc='Pre-compute random walks', leave=False):
        rws[(u, v)][u] = walker.walk_from(
            graph=g,
            start_node=u,
            forbidden_edge=(u, v),
        )
        rws[(u, v)][v] = walker.walk_from(
            graph=g,
            start_node=v,
            forbidden_edge=(u, v),
        )

    return rws
