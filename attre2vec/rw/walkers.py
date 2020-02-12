"""Implementations of different random walkers."""
import abc
import random
from typing import List, Tuple

import networkx as nx


class BaseWalker(abc.ABC):
    """Base class for random walker."""

    def __init__(self, walk_length: int, num_walks: int):
        """Inits BaseWalker."""
        self._walk_length = walk_length
        self._num_walks = num_walks

    @abc.abstractmethod
    def walk_from(
        self,
        graph: nx.Graph,
        start_node: int,
        forbidden_edge: Tuple[int, int],
    ) -> List[Tuple[int, int, int]]:
        """Performs random walk from given node."""
        pass


class UniformRandomWalker(BaseWalker):
    """Random walker that samples neighbors from uniform distribution."""

    def walk_from(
        self,
        graph: nx.Graph,
        start_node: int,
        forbidden_edge: Tuple[int, int],
    ) -> List[List[Tuple[int, int]]]:
        """Performs random walks from given node."""
        u, v = forbidden_edge
        forbidden_edges = [(u, v), (v, u)]

        walks = []
        for _ in range(self._num_walks):
            walk = []
            curr_node = start_node

            while len(walk) != self._walk_length:
                neigh_edges = [
                    e
                    for e in graph.edges(curr_node)
                    if e not in forbidden_edges
                ]

                if not neigh_edges:
                    break

                edge = random.choice(neigh_edges)
                walk.append(edge)

                curr_node = edge[1]

            walk = walk + [(-1, -1)] * max(0, self._walk_length - len(walk))
            walks.append(walk)

        return walks


def make_walker(rw_cfg: dict) -> BaseWalker:
    """Creates random walker based on config."""
    walkers = {
        'uniform': UniformRandomWalker,
    }

    walker_name = rw_cfg['walker-name']

    if walker_name not in walkers.keys():
        raise RuntimeError(f'No such random walker: {walker_name}')

    walker_cls = walkers[walker_name]

    return walker_cls(**rw_cfg['args'])
