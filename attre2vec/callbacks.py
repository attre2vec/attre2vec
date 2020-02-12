"""Implementations of custom callbacks for PyTorch Lightning."""
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as ptl


class LossAccumulator(ptl.Callback):
    """Custom callback for accumulating loss values."""

    def __init__(self, filepath):
        """Inits LossAccumulator."""
        self._path = filepath
        self._loss_values = {
            'total': {'train': [], 'val': []},
            'rec': {'train': [], 'val': []},
            'emb': {'train': [], 'val': []},
        }
        self._metrics = {
            'auc': [],
            'acc': [],
            'f1': [],
        }

    def on_epoch_end(self, trainer, pl_module):
        """Extracts loss value on end of each epoch."""
        logs = trainer.callback_metrics

        total_loss = self._loss_values['total']
        total_loss['train'].append(logs['train_loss'].cpu().item())
        total_loss['val'].append(logs['val_loss'])

        rec_loss = self._loss_values['rec']
        rec_loss['train'].append(logs['rec_loss'])
        rec_loss['val'].append(logs['val_rec_loss'])

        emb_loss = self._loss_values['emb']
        emb_loss['train'].append(logs['emb_loss'])
        emb_loss['val'].append(logs['val_emb_loss'])

        self._metrics['auc'].append(logs['auc'])
        self._metrics['acc'].append(logs['acc'])
        self._metrics['f1'].append(logs['f1'])

        self.save(self._path)

    def save(self, path):
        """Creates plot with loss values and saves it to the given path."""
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20, 20))

        y_auc = self._metrics['auc']
        y_acc = self._metrics['acc']
        y_f1 = self._metrics['f1']

        x = list(range(1, len(y_auc) + 1))

        cp = sns.color_palette(n_colors=5)

        # Losses (total)
        tot_tr = self._loss_values['total']['train']
        tot_val = self._loss_values['total']['val']

        self._plot_metric(ax=axs[0], x=x, y=tot_tr, color=cp[0], label='Train')
        self._plot_metric(ax=axs[0], x=x, y=tot_val, color=cp[1], label='Val')

        axs[0].set(
            ylim=(0, max(*tot_tr, *tot_val)),
            title='Total losses over epochs',
            ylabel='Loss',
            xlabel='Epoch',
        )
        axs[0].legend()

        # Losses (rec)
        rec_tr = self._loss_values['rec']['train']
        rec_val = self._loss_values['rec']['val']

        self._plot_metric(ax=axs[1], x=x, y=rec_tr, color=cp[0], label='Train')
        self._plot_metric(ax=axs[1], x=x, y=rec_val, color=cp[1], label='Val')

        axs[1].set(
            ylim=(0, max(*rec_tr, *rec_val)),
            title='Reconstruction losses over epochs',
            ylabel='Loss',
            xlabel='Epoch',
        )
        axs[1].legend()

        # Losses (emb)
        emb_tr = self._loss_values['emb']['train']
        emb_val = self._loss_values['emb']['val']

        self._plot_metric(ax=axs[2], x=x, y=emb_tr, color=cp[0], label='Train')
        self._plot_metric(ax=axs[2], x=x, y=emb_val, color=cp[1], label='Val')

        axs[2].set(
            ylim=(0, max(*emb_tr, *emb_val)),
            title='Embedding losses over epochs',
            ylabel='Loss',
            xlabel='Epoch',
        )
        axs[2].legend()

        # Metrics
        self._plot_metric(ax=axs[3], x=x, y=y_auc, color=cp[2], label='AUC')
        self._plot_metric(ax=axs[3], x=x, y=y_acc, color=cp[3], label='Acc')
        self._plot_metric(ax=axs[3], x=x, y=y_f1, color=cp[4], label='F1')

        axs[3].set(
            ylim=(0, 1),
            title='Metrics over epochs',
            ylabel='Metric values',
            xlabel='Epoch',
        )
        axs[3].legend()

        fig.tight_layout()
        fig.savefig(path)
        plt.close()

    @staticmethod
    def _plot_metric(ax, x, y, color, label):
        ax.plot(x, y, linestyle='--', alpha=0.5, color=color)
        ax.plot(x, y, linestyle='', marker='x', color=color, label=label)


class CustomModelCheckpoint(ptl.callbacks.ModelCheckpoint):
    """Implementation of model checkpoint for custom paths."""

    def on_validation_end(self, trainer, pl_module):
        """Saves model if monitored metrics has improved."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        if (
            (self.epoch_last_check is None)
            or (epoch - self.epoch_last_check >= self.period)
        ):
            self.epoch_last_check = epoch

            filepath = os.path.join(self.dirpath, self.filename)

            if self.save_top_k != -1:
                current = metrics.get(self.monitor)

                if self.check_monitor_top_k(current):
                    self._do_check_save(filepath, current, epoch)
