import pickle

import networkx as nx
import numpy as np
import torch


for name in ('cora', 'citeseer', 'pubmed'):
    with open(f'data/datasets/{name}.pkl', 'rb') as fin:
        dataset = pickle.load(fin)

    test_graph = dataset['original_graph']
    e2i = dataset['edge2idx']
    H = dataset['H']

    node_fts = torch.zeros((test_graph.number_of_nodes(), 128))

    for u, v in test_graph.edges():
        ef = H[e2i[(u, v)]][3:-1]

        node_fts[u] = ef[:128]
        node_fts[v] = ef[128:]

    train_nodes = []
    for idx in range(dataset['num_datasets']):
        tn = []
        for u, v in dataset['Xy'][idx]['train']['X']:
            if u not in tn:
                tn.append(u)

            if v not in tn:
                tn.append(v)

        train_nodes.append(tn)

    nx.write_edgelist(test_graph, f'GraphSAGE/data/{name}.edgelist')
    np.save(f'GraphSAGE/data/{name}-node-features', node_fts.numpy())
    with open(f'GraphSAGE/data/{name}-train-nodes.pkl', 'wb') as fout:
        pickle.dump(train_nodes, fout)
