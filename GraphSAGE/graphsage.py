import os
import time
from typing import Any, Dict, Optional, Sequence

import networkx as nx
import numpy as np
import tensorflow as tf

from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.models import (
    BipartiteEdgePredLayer,
    GeneralizedModel,
    MeanAggregator,
    SAGEInfo,
)
from graphsage.neigh_samplers import UniformNeighborSampler


class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(
        self,
        placeholders,
        features,
        degrees,
        layer_infos,
        aggregators: Optional[Sequence[MeanAggregator]] = None,
        lr: float = 1e-5,
        neg_sample_size: int = 25,
        concat: bool = True,
        model_size: str = "small",
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
                        NOTE: Pass a None object to train in featureless mode
                        (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random
                re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the
                parameters of all the recursive layers. See SAGEInfo definition
                above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
        """
        super().__init__(**kwargs)
        self.aggregator_cls = MeanAggregator
        self.aggregators = aggregators

        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.model_size = model_size
        self.embeds = None
        self.features = tf.Variable(
            tf.constant(features, dtype=tf.float32), trainable=False
        )
        self.degrees = degrees
        self.concat = concat

        self.dims = [0 if features is None else features.shape[1]]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))]
        )
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.neg_sample_size = neg_sample_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.grad = None
        self.build()

    def sample(self, inputs, layer_infos, batch_size=None):
        """ Sample neighbors to be the supportive fields for multi-layer
            convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and
                negative samples).
        """

        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(
        self,
        samples,
        input_features,
        dims,
        num_samples,
        support_sizes,
        batch_size=None,
        aggregators=None,
        name=None,
        concat=False,
        model_size="small",
    ):
        """ At each layer, aggregate hidden representations of neighbors to
            compute the hidden representations at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at
                each layer of the network. Length is the number of layers + 1.
                Each is a vector of node indices.
            input_features: the input features for each sample of various hops
                away.
            dims: a list of dimensions of the hidden representations from the
                input layer to the final layer. Length is the number of
                layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for
                each layer.
            batch_size: the number of inputs (different for batch inputs and
                negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """

        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [
            tf.nn.embedding_lookup(input_features, node_samples)
            for node_samples in samples
        ]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(
                        dim_mult * dims[layer],
                        dims[layer + 1],
                        act=lambda x: x,
                        dropout=self.placeholders["dropout"],
                        name=name,
                        concat=concat,
                        model_size=model_size,
                    )
                else:
                    aggregator = self.aggregator_cls(
                        dim_mult * dims[layer],
                        dims[layer + 1],
                        dropout=self.placeholders["dropout"],
                        name=name,
                        concat=concat,
                        model_size=model_size,
                    )
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes
            # that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [
                    batch_size * support_sizes[hop],
                    num_samples[len(num_samples) - hop - 1],
                    dim_mult * dims[layer],
                ]
                h = aggregator(
                    (hidden[hop], tf.reshape(hidden[hop + 1], neigh_dims))
                )
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(
            tf.cast(self.placeholders["batch2"], dtype=tf.int64),
            [self.batch_size, 1],
        )
        self.neg_samples, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist(),
        )

        # perform "convolution"
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos
        ]
        self.outputs1, self.aggregators = self.aggregate(
            samples1,
            [self.features],
            self.dims,
            num_samples,
            support_sizes1,
            aggregators=self.aggregators,
            concat=self.concat,
            model_size=self.model_size,
        )
        self.outputs2, _ = self.aggregate(
            samples2,
            [self.features],
            self.dims,
            num_samples,
            support_sizes2,
            aggregators=self.aggregators,
            concat=self.concat,
            model_size=self.model_size,
        )

        neg_samples, neg_support_sizes = self.sample(
            self.neg_samples, self.layer_infos, self.neg_sample_size
        )
        self.neg_outputs, _ = self.aggregate(
            neg_samples,
            [self.features],
            self.dims,
            num_samples,
            neg_support_sizes,
            batch_size=self.neg_sample_size,
            aggregators=self.aggregators,
            concat=self.concat,
            model_size=self.model_size,
        )

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(
            dim_mult * self.dims[-1],
            dim_mult * self.dims[-1],
            self.placeholders,
            act=tf.nn.sigmoid,
            bilinear_weights=False,
            name="edge_predict",
        )

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
        self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [
            (
                tf.clip_by_value(grad, -5.0, 5.0)
                if grad is not None
                else None,
                var,
            )
            for grad, var in grads_and_vars
        ]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(
            self.outputs1, self.outputs2, self.neg_outputs
        )
        tf.summary.scalar("loss", self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.outputs1, self.neg_outputs
        )
        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, self.neg_sample_size]
        )
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32))
        )
        tf.summary.scalar("mrr", self.mrr)


class GraphSAGEAdapter:
    def __init__(
        self,
        model_size: str = "small",
        dropout: float = 0.0,
        *,
        max_degree: int = 100,
        dim_1: int = 128,
        dim_2: int = 128,
        samples_1: int = 25,
        samples_2: int = 10,
    ):

        self.placeholders = {}
        self.model_size = model_size
        self.dropout = dropout

        self.max_degree = max_degree
        self.dim_1 = dim_1
        self.dim_2 = dim_2

        self.samples_1 = samples_1
        self.samples_2 = samples_2

        self.session: Optional[tf.Session] = None
        self.model: Optional[SampleAndAggregate] = None
        self.minibatch_iter: Optional[EdgeMinibatchIterator] = None
        self.construct_placeholders()

    def construct_placeholders(self):
        # Define placeholders
        self.placeholders = {
            "batch1": tf.placeholder(tf.int32, shape=(None,), name="batch1"),
            "batch2": tf.placeholder(tf.int32, shape=(None,), name="batch2"),
            # negative samples for all nodes in the batch
            "neg_samples": tf.placeholder(
                tf.int32, shape=(None,), name="neg_sample_size"
            ),
            "dropout": tf.placeholder_with_default(
                0.0, shape=(), name="dropout"
            ),
            "batch_size": tf.placeholder(tf.int32, name="batch_size"),
        }

    def fit(
        self,
        graph: nx.Graph,
        features: np.ndarray,
        id_map: Optional[Dict[Any, int]],
        epochs: int = 1,
        batch_size: int = 512,
        lr: float = 0.00001,
        *,
        base_log_dir: Optional[str] = None,
        print_every: int = 50,
        max_total_steps: int = 10 ** 10,
    ):
        if features is not None:
            features = np.vstack([features, np.zeros((features.shape[1],))])

        if id_map is None:
            id_map = {an_id: an_id for an_id in graph.nodes()}

        self.minibatch_iter = EdgeMinibatchIterator(
            graph,
            id_map,
            self.placeholders,
            batch_size=batch_size,
            max_degree=self.max_degree,
        )

        adj_info_ph = tf.placeholder(
            tf.int32, shape=self.minibatch_iter.adj.shape
        )
        adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [
            SAGEInfo("node", sampler, self.samples_1, self.dim_1),
            SAGEInfo("node", sampler, self.samples_2, self.dim_2),
        ]

        self.model = SampleAndAggregate(
            self.placeholders,
            features,
            self.minibatch_iter.deg,
            layer_infos=layer_infos,
            model_size=self.model_size,
            logging=True,
        )

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        self.session = tf.Session(config=config)

        # Init variables
        self.session.run(
            tf.global_variables_initializer(),
            feed_dict={adj_info_ph: self.minibatch_iter.adj},
        )

        merged = tf.summary.merge_all()
        summary_writer = None
        if base_log_dir is not None:
            summary_writer = tf.summary.FileWriter(
                self._log_dir(base_log_dir, lr), self.session.graph
            )

        train_shadow_mrr = None

        total_steps = 0
        avg_time = 0.0

        # Train model
        for epoch in range(epochs):
            self.minibatch_iter.shuffle()

            iteration = 0
            print("Epoch: %04d" % (epoch + 1))
            while not self.minibatch_iter.end():
                # Construct feed dictionary
                feed_dict = self.minibatch_iter.next_minibatch_feed_dict()
                feed_dict.update({self.placeholders["dropout"]: self.dropout})

                start = time.time()
                # Training step
                outs = self.session.run(
                    [
                        merged,
                        self.model.opt_op,
                        self.model.loss,
                        self.model.ranks,
                        self.model.aff_all,
                        self.model.mrr,
                        self.model.outputs1,
                    ],
                    feed_dict=feed_dict,
                )
                train_cost = outs[2]
                train_mrr = outs[5]
                if train_shadow_mrr is None:
                    train_shadow_mrr = train_mrr  #
                else:
                    train_shadow_mrr -= (1 - 0.99) * (
                        train_shadow_mrr - train_mrr
                    )

                if (
                    total_steps % print_every == 0
                    and summary_writer is not None
                ):
                    summary_writer.add_summary(outs[0], total_steps)

                # Print results
                avg_time = (avg_time * total_steps + time.time() - start) / (
                    total_steps + 1
                )

                if total_steps % print_every == 0:
                    print(
                        "Iter:",
                        "%04d" % iteration,
                        "train_loss=",
                        "{:.5f}".format(train_cost),
                        "train_mrr=",
                        "{:.5f}".format(train_mrr),
                        "train_mrr_ema=",
                        "{:.5f}".format(train_shadow_mrr),
                        # exponential moving average
                        "time=",
                        "{:.5f}".format(avg_time),
                    )

                iteration += 1
                total_steps += 1

                if total_steps > max_total_steps:
                    break

            if total_steps > max_total_steps:
                break

        print("Optimization Finished!")

    def _log_dir(self, base_log_dir: str, lr: float):
        log_dir = base_log_dir + "/unsup"
        log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model="graphsage", model_size=self.model_size, lr=lr
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def predict(self, batch_size: int) -> np.ndarray:
        val_embeddings = []
        finished = False
        seen = set([])
        nodes = []
        iter_num = 0
        while not finished:
            (
                feed_dict_val,
                finished,
                edges,
            #) = self.minibatch_iter.incremental_embed_feed_dict_val_only(
            ) = self.minibatch_iter.incremental_embed_feed_dict(
                batch_size, iter_num
            )
            iter_num += 1
            outs_val = self.session.run(
                [self.model.loss, self.model.mrr, self.model.outputs1],
                feed_dict=feed_dict_val,
            )
            # ONLY SAVE FOR embeds1 because of planetoid
            for i, edge in enumerate(edges):
                if not edge[0] in seen:
                    val_embeddings.append(outs_val[-1][i, :])
                    nodes.append(edge[0])
                    seen.add(edge[0])
        val_embeddings = np.vstack(val_embeddings)
        return val_embeddings


def main():
    import pickle
    from tqdm import tqdm

    for name in ('cora', 'citeseer', 'pubmed'):
        for idx in tqdm(range(10), desc=f'{name}'):
            graph = nx.read_edgelist(f'data/{name}.edgelist', nodetype=int)
            features = np.load(f'data/{name}-node-features.npy')

            with open(f'data/{name}-train-nodes.pkl', 'rb') as fin:
                train_nodes = pickle.load(fin)[idx]

            for u in graph.nodes():
                is_test = u not in train_nodes

                graph.node[u]['test'] = is_test
                graph.node[u]['val'] = is_test

            for u, v in graph.edges():
                if (
                        graph.node[u]['val'] or graph.node[v]['val'] or
                        graph.node[u]['test'] or graph.node[v]['test']
                ):
                    graph[u][v]['train_removed'] = True
                else:
                    graph[u][v]['train_removed'] = False

            saga = GraphSAGEAdapter(dim_1=32, dim_2=32)
            saga.fit(
                graph=graph,
                features=features,
                id_map=None,
                epochs=10,
                batch_size=32,
                max_total_steps=200,
            )

            embs = saga.predict(batch_size=32)

            tf.reset_default_graph()

            np.save(f'n_embed/{name}-{idx}', embs)


if __name__ == "__main__":
    main()
