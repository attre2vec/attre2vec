"""Implementation of AttrE2vec model."""
import importlib
import json

import pytorch_lightning as ptl
import torch

from attre2vec.datasets import edge_dataset as eds
from attre2vec.layers import edge_sage as ng_es
from attre2vec.layers.task import decoders as ng_dec
from attre2vec import sampling as ng_sam
from attre2vec import utils as ng_util


def make_clf(path, args):
    """Creates a classifier given config."""
    module, model = path.rsplit('.', maxsplit=1)
    clf_class = getattr(importlib.import_module(module), model)

    clf = clf_class(**args)
    return clf


class UnsupervisedModel(ptl.LightningModule):
    """Abstract class for unsupervised models."""

    def __init__(self, hparams, **kwargs):
        """Inits UnsupervisedModel."""
        super().__init__(**kwargs)

        self.hparams = hparams

        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self._es = ng_es.EdgeSampleAggregate.from_cfg(hparams=self.hparams)
        self._es.to(self._device)

        self._decoder = ng_dec.make_decoder(
            name=self.hparams.decoder,
            hparams=self.hparams,
        )
        self._decoder.to(self._device)

        self._rec_loss = torch.nn.MSELoss()
        self._emb_loss = torch.nn.CosineEmbeddingLoss()

        self.base_clf = None
        self._bc_cfg = json.loads(self.hparams.base_clf_cfg)
        self._bc_data = None

    def set_base_clf_data(self, train_data, val_data):
        """Initializes the data used for training and testing the base clf."""
        self._bc_data = {
            'train': train_data,
            'val': val_data,
        }

    def forward(self, edge_data: eds.EdgeData, training=False):
        """Performs forward pass of model."""
        # Aggregate and encode
        h_xe_0 = edge_data.h[[
            edge_data.edge2idx[tuple(e)]
            for e in edge_data.xe.tolist()
        ]].to(self._device)

        h_xe, meta = self._es(
            xe=edge_data.xe,
            h_xe=h_xe_0,
            h=edge_data.h,
            m=edge_data.m,
            rws_idx=edge_data.rws_idx,
        )

        if training:
            return h_xe, meta

        return h_xe

    def _step(self, edge_data: eds.EdgeData):
        """Executes train/val step."""
        # Positive/negative sampling
        ed = edge_data
        original_xe = ed.xe

        samples = ng_sam.precompute_samples(
            xe=[tuple(e) for e in original_xe.tolist()],
            rws=ed.rws, graph=ed.graph,
            num_pos=self.hparams.samples_pos,
            num_neg=self.hparams.samples_neg,
        )

        xe_plus = samples['plus']['xe']
        idxs_plus = samples['plus']['idxs']

        xe_minus = samples['minus']['xe']
        idxs_minus = samples['minus']['idxs']

        # Make forward
        ed.xe = torch.cat([
            original_xe,
            torch.tensor(xe_plus, device=self._device),
            torch.tensor(xe_minus, device=self._device)
        ], dim=0)
        xe_all, meta = self.forward(ed, training=True)

        os, ps = original_xe.size(0), xe_plus.shape[0]
        h_xe = xe_all[:os]
        h_xe_plus = xe_all[os:os + ps]
        h_xe_minus = xe_all[os + ps:]

        # Reconstruct features
        h_xe_0 = ed.h[[
            ed.edge2idx[tuple(e)]
            for e in original_xe.tolist()
        ]].to(self._device)

        h_xe_rec = self._decoder(h_xe)

        # Compute loss
        x1 = h_xe[[*idxs_plus, *idxs_minus]]
        x2 = torch.cat([h_xe_plus, h_xe_minus], dim=0)

        target = torch.cat([
            torch.ones(h_xe_plus.size(0), device=self._device),
            (-1) * torch.ones(h_xe_minus.size(0), device=self._device),
        ], dim=0)

        mixing = self.hparams.mixing
        emb_loss = self._emb_loss(input1=x1, input2=x2, target=target)
        rec_loss = self._rec_loss(input=h_xe_rec, target=h_xe_0)

        loss = mixing * emb_loss + (1 - mixing) * rec_loss

        edge_data.xe = original_xe

        meta['losses'] = {
            'rec': rec_loss.detach().item(),
            'emb': emb_loss.detach().item(),
        }

        return loss, meta

    def configure_optimizers(self):
        """Configures optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            params=[
                *self._es.parameters(),
                *self._decoder.parameters(),
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        """Executes training step."""
        loss, meta = self._step(edge_data=batch)

        # Results dict
        rd = {
            'train_loss': loss,
            'rec_loss': meta['losses']['rec'],
            'emb_loss': meta['losses']['emb'],
        }

        return {'loss': loss, 'progress_bar': {'train_loss': loss}, 'log': rd}

    def validation_step(self, batch, batch_idx):
        """Executes validation step."""
        if batch_idx == 0:  # After switching from train to val
            self.train_base_clf()

        val_loss, meta = self._step(edge_data=batch)

        # Log alpha values
        tb = self.logger.experiment
        for idx, name in enumerate(('h_0', 'f_u', 'f_v')):
            tb.add_histogram(f'alpha-{name}', meta['alpha'][:, idx, :])

        return {
            'val_loss': val_loss,
            'val_rec_loss': meta['losses']['rec'],
            'val_emb_loss': meta['losses']['emb'],
        }

    def validation_epoch_end(self, outputs):
        """Calculates mean validation loss in each epoch."""
        val_loss = torch.mean(torch.tensor([
            o['val_loss'] for o in outputs
        ])).item()

        rec_loss = torch.mean(torch.tensor([
            o['val_rec_loss'] for o in outputs
        ])).item()

        emb_loss = torch.mean(torch.tensor([
            o['val_emb_loss'] for o in outputs
        ])).item()

        # Log parameter values
        tb = self.logger.experiment
        for name, param in self.named_parameters():
            tb.add_histogram(f'VAL-{name}', param)

        # Check classification ability
        mtrs = self.eval_on_base_clf(data=self._bc_data['val'])

        # Results dict
        rd = {
            'val_loss': val_loss,
            'val_rec_loss': rec_loss,
            'val_emb_loss': emb_loss,
            'auc': mtrs['auc'],
            'acc': mtrs['accuracy'],
            'f1': mtrs['macro avg']['f1-score'],
        }

        return {**rd, 'progress_bar': rd, 'log': rd}

    def data_to_emb_labels(self, data):
        """Converts edge data to embeddings and their labels."""
        emb = torch.cat([self.forward(ed).cpu() for ed in data], dim=0)
        labels = torch.cat([ed.y.cpu() for ed in data], dim=0)
        return emb, labels

    def train_base_clf(self):
        """Trains the logistic regression classifier (validation purposes)."""
        train_h, train_y = self.data_to_emb_labels(data=self._bc_data['train'])
        clf = make_clf(path=self._bc_cfg['path'], args=self._bc_cfg['args'])
        clf.fit(train_h, train_y)

        self.base_clf = clf

    def eval_on_base_clf(self, data):
        """Uses fitted base classifier to get metrics."""
        emb, y_true = self.data_to_emb_labels(data=data)
        y_score = self.base_clf.predict_proba(emb)
        y_pred = self.base_clf.predict(emb)

        mtrs = ng_util.calc_metrics(
            y_score=y_score,
            y_pred=y_pred,
            y_true=y_true,
            max_cls=self.hparams.num_cls,
        )

        return mtrs

    def on_save_checkpoint(self, checkpoint):
        """Adds base classifier when saving checkpoint."""
        checkpoint['base_clf'] = self.base_clf

    def on_load_checkpoint(self, checkpoint):
        """Retrieves base classifier when loading checkpoint."""
        self.base_clf = checkpoint['base_clf']
