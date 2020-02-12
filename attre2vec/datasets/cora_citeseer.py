"""Code for preprocessing Cora/Citeseer/Pubmed datasets."""
import os

from gensim.models import doc2vec
import networkx as nx
import node2vec as n2v
import numpy as np
import torch
from tqdm.auto import tqdm


class InitialEdgeFeatureExtractor:
    """Extracts cos sim of node vectors, word ratios and concat of Doc2vec."""

    def __init__(self):
        """Inits InitialEdgeFeatureExtractor."""
        self._cos = torch.nn.CosineSimilarity()

    def extract(
        self,
        g,
        raw_node_features,
        d2v_node_features,
        node_classes,
        verbose,
    ):
        """Implements feature extraction."""
        original_edges = sorted(g.edges())
        max_cls = node_classes.max().item()

        edge_fts = {}
        edge_cls = []
        for u, v in tqdm(original_edges,
                         desc='Extracting edge features',
                         disable=not verbose,
                         leave=False):
            raw_x_u = raw_node_features[u].float()
            x_u = d2v_node_features[u].float()
            c_u = node_classes[u].item()

            raw_x_v = raw_node_features[v].float()
            x_v = d2v_node_features[v].float()
            c_v = node_classes[v].item()

            # Add original edge
            edge_fts[(u, v)] = self._mk_features(
                raw_x_u=raw_x_u, raw_x_v=raw_x_v,
                x_u=x_u, x_v=x_v,
                is_original_edge=1,
            )
            edge_cls.append(c_u if c_u == c_v else max_cls + 1)

            # Add reversed edge (for aggregation models)
            edge_fts[(v, u)] = self._mk_features(
                raw_x_u=raw_x_v, raw_x_v=raw_x_u,
                x_u=x_v, x_v=x_u,
                is_original_edge=0,
            )

        H, edge2idx = self._to_edge_feature_tensor(edge_fts)

        return original_edges, edge_cls, H, edge2idx

    def _mk_features(self, raw_x_u, raw_x_v, x_u, x_v, is_original_edge):
        return [
            self._cos_sim(raw_x_u, raw_x_v),
            self._unique_word_ratio(raw_x_u),
            self._unique_word_ratio(raw_x_v),
            *self._concat(x_u, x_v),
            is_original_edge,
        ]

    def _cos_sim(self, a, b):
        return self._cos(a.unsqueeze(0), b.unsqueeze(0)).item()

    @staticmethod
    def _unique_word_ratio(a):
        return torch.sum(a).item() / a.size(0)

    @staticmethod
    def _concat(a, b):
        return torch.cat([a, b], dim=-1).tolist()

    @staticmethod
    def _to_edge_feature_tensor(ef: dict):
        edges = sorted(ef.keys())
        idxs = range(len(ef.keys()))

        edge2idx = dict(zip(edges, idxs))
        idx2edge = dict(zip(idxs, edges))

        H = torch.tensor([ef[idx2edge[idx]] for idx in idxs], dtype=torch.float)

        H = torch.cat([torch.zeros(1, H.size(1)), H], dim=0)

        edge2idx = {k: v + 1 for k, v in edge2idx.items()}
        edge2idx[(-1, -1)] = 0

        return H, edge2idx


class Node2vecExtractor:
    """Extracts node features using Node2vec."""

    def __init__(self, dim):
        """Inits Node2vecExtractor."""
        self._dim = dim

    def extract(self, g):
        """Implements feature extraction."""
        emb = n2v.Node2Vec(
            graph=g,
            dimensions=self._dim,
            workers=8,
            p=4, q=1,
            quiet=True,
        ).fit()

        nodes = sorted(g.nodes())
        idxs = range(g.number_of_nodes())

        node2idx = dict(zip(nodes, idxs))
        idx2node = dict(zip(idxs, nodes))

        M = torch.tensor([emb.wv[str(idx2node[idx])] for idx in idxs])

        M = torch.cat([torch.zeros(1, M.size(1)), M], dim=0)

        node2idx = {k: v + 1 for k, v in node2idx.items()}
        node2idx[-1] = 0

        return {'M': M, 'node2idx': node2idx}


def read_node_info(path):
    """Reads node features and classes from raw file."""
    raw = {}

    with open(path, 'r') as fin:
        for line in fin.readlines():
            row = line.strip().split('\t')
            node_id = row[0]
            fts = [int(f) for f in row[1:-1]]
            cls = row[-1]

            raw[node_id] = {'fts': fts, 'cls': cls}

    unique_cls = set(v['cls'] for v in raw.values())
    orig2new_cls = dict(zip(
        sorted(unique_cls),
        range(len(unique_cls))
    ))

    for node_info in raw.values():
        node_info['cls'] = orig2new_cls[node_info['cls']]

    return raw


def read_graph(path):
    """Reads the citation links and builds graphs from it."""
    cites = []
    with open(path, 'r') as fin:
        for line in tqdm(fin.readlines(), desc='Read raw graph', leave=False):
            e = line.strip().split('\t')
            u, v = e[0], e[1]

            if u == v:  # Remove self-loops
                continue

            if (v, u) in cites:
                continue

            cites.append((u, v))

    g = nx.DiGraph()
    g.add_edges_from(cites)

    giant_component_nodes = next(nx.connected_components(G=g.to_undirected()))
    g = g.subgraph(nodes=giant_component_nodes)

    return g


def remove_unknown_nodes(g, node_info):
    """Removes nodes from graph that are not present in `content` file."""
    g_cpy = g.copy()

    node_to_remove = [
        node
        for node in g_cpy.nodes()
        if node not in node_info.keys()
    ]

    g_cpy.remove_nodes_from(node_to_remove)

    while True:
        zero_deg_nodes = [node for node, deg in g_cpy.degree() if deg == 0]
        if not zero_deg_nodes:
            break

        g_cpy.remove_nodes_from(zero_deg_nodes)

    return g_cpy


def read_raw_data(path):
    """Reads the raw graph data and cleans up unknown nodes."""
    raw_graph = read_graph(path=os.path.join(path, 'cites'))

    raw_node_info = read_node_info(path=os.path.join(path, 'content'))

    g = remove_unknown_nodes(g=raw_graph, node_info=raw_node_info)

    orig2new_nodes = dict(zip(
        sorted(g.nodes()),
        range(g.number_of_nodes())
    ))
    new2orig_nodes = dict(zip(
        range(g.number_of_nodes()),
        sorted(g.nodes()),
    ))

    g = nx.relabel_nodes(g, mapping=orig2new_nodes)

    node_fts = []
    node_cls = []
    for node_id in sorted(new2orig_nodes.keys()):
        orig_node_id = new2orig_nodes[node_id]

        node_fts.append(raw_node_info[orig_node_id]['fts'])
        node_cls.append(raw_node_info[orig_node_id]['cls'])

    node_fts = torch.tensor(node_fts)
    node_cls = torch.tensor(node_cls)

    return g, node_fts, node_cls


def to_doc2vec_emb(node_fts, dim, epochs):
    """Computes Doc2vec embeddings from node BoW features."""
    documents = [
        doc2vec.TaggedDocument(
            words=[str(idx) for idx in range(len(doc)) if doc[idx] == 1],
            tags=[f'Doc_{doc_idx}'],
        )
        for doc_idx, doc in enumerate(node_fts.tolist())
    ]
    d2v = doc2vec.Doc2Vec(dm=0, vector_size=dim, workers=8)
    d2v.build_vocab(documents=documents)
    d2v.train(documents=documents, total_examples=len(documents), epochs=epochs)

    vecs = torch.tensor([
        d2v.docvecs[f'Doc_{idx}']
        for idx in range(node_fts.size(0))
    ])
    return vecs


def sample_train_val_test(X, y, num_cls, train_size, val_size, test_size):
    """Samples train/val/test splits."""
    # Train
    X_train, y_train = [], []
    for cls in range(num_cls):
        xc, yc = X[y == cls], y[y == cls]
        idxs = np.random.choice(
            range(xc.shape[0]),
            size=train_size,
            replace=False,
        )

        X_train.extend([tuple(x) for x in xc[idxs].tolist()])
        y_train.extend(yc[idxs].tolist())

    # Val
    rest_idxs = [i for i in range(X.shape[0]) if tuple(X[i]) not in X_train]

    val_idxs = np.random.choice(rest_idxs, size=val_size, replace=False)
    X_val = [tuple(x) for x in X[val_idxs].tolist()]
    y_val = y[val_idxs].tolist()

    # Test
    rest_idxs = [
        i for i in range(X.shape[0])
        if tuple(X[i]) not in X_train and tuple(X[i]) not in X_val
    ]

    test_idxs = np.random.choice(rest_idxs, size=test_size, replace=False)
    X_test = [tuple(x) for x in X[test_idxs].tolist()]
    y_test = y[test_idxs].tolist()

    return {
        'train': {
            'X': X_train,
            'y': y_train,
        },
        'val': {
            'X': X_val,
            'y': y_val,
        },
        'test': {
            'X': X_test,
            'y': y_test,
        }
    }


def read_cora_citeseer_pubmed(
    path: str,
    node_dim: int,
    doc2vec_kwargs: dict,
    split_sizes: dict,
    num_datasets: int,
    verbose: bool = False,
):
    """Reads and preprocesses Cora dataset."""
    g, raw_node_fts, node_cls = read_raw_data(path)

    # Convert node BoW features to Doc2vec embeddings
    d2v_node_fts = to_doc2vec_emb(
        node_fts=raw_node_fts,
        dim=doc2vec_kwargs['dim'],
        epochs=doc2vec_kwargs['epochs'],
    )

    # Extract edge and new node features
    edges, edge_labels, H, edge2idx = InitialEdgeFeatureExtractor().extract(
        g=g,
        raw_node_features=raw_node_fts,
        d2v_node_features=d2v_node_fts,
        node_classes=node_cls,
        verbose=verbose,
    )
    num_cls = (node_cls.max() + 2).item()

    # Sample multiple datasets
    Xy, graphs = [], []
    M = []
    for _ in tqdm(range(num_datasets), desc='Datasets'):
        # Train/val/test split
        tr_val_te = sample_train_val_test(
            X=np.array(edges),
            y=np.array(edge_labels),
            num_cls=num_cls,
            train_size=split_sizes['train'],
            val_size=split_sizes['validation'],
            test_size=split_sizes['test'],
        )
        Xy.append(tr_val_te)

        # Remove test edges from graph (inductive)
        g_cpy = g.copy()
        g_cpy.remove_edges_from(tr_val_te['test']['X'])
        g_cpy.remove_nodes_from([n for n, d in g_cpy.degree() if d == 0])
        graphs.append(g_cpy)

        # Compute Node2vec features
        M.append(Node2vecExtractor(dim=node_dim).extract(g_cpy))

    M_test = Node2vecExtractor(dim=node_dim).extract(g)

    return {
        # Dataset independent
        'original_graph': g,
        'H': H,
        'edge2idx': edge2idx,
        'num_cls': num_cls,
        'dims': {'node': node_dim, 'edge': H.size(1)},
        'num_datasets': num_datasets,

        # Dataset dependent
        'Xy': Xy,
        'graphs': graphs,
        'M': M,
        'M_test': M_test,
    }
