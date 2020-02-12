"""Script for training AttrE2vec model."""
import argparse
import json
import logging
import os
import pickle
import yaml

import pytorch_lightning as ptl
from pytorch_lightning import callbacks as ptl_cb
from pytorch_lightning import loggers as ptl_log
import torch
from tqdm import tqdm

from attre2vec import callbacks as ae_cb
from attre2vec.datasets import edge_dataset as eds
from attre2vec import model as ae_mod


def get_args() -> argparse.Namespace:
    """Gets arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-cc',
        '--common-config',
        required=True,
        help='Path to common config file',
    )
    parser.add_argument(
        '-mc',
        '--model-config',
        required=True,
        help='Path to model config file',
    )
    return parser.parse_args()


def main():
    """Runs the script."""
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    with open(args.common_config, 'r') as fin:
        common_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    with open(args.model_config, 'r') as fin:
        model_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    dataset_path = common_cfg['paths']['input']['dataset']
    rws_path = common_cfg['paths']['input']['rws']

    model_path = common_cfg['paths']['output']['model']
    losses_path = common_cfg['paths']['output']['losses']
    logs_path = common_cfg['paths']['output']['logs']
    metrics_path = common_cfg['paths']['output']['metrics']
    vectors_path = common_cfg['paths']['output']['vectors']

    base_clf_cfg = common_cfg['base_clf']

    num_epochs = model_cfg['training']['epochs']

    # Datasets samples
    datasets = eds.EdgeDatasets(dataset_path=dataset_path, rws_path=rws_path)

    # Hyperparameters
    emb_dim = model_cfg['args'].pop('emb-dim')
    hparams = argparse.Namespace(
        **model_cfg['args'],
        **model_cfg['training'],
        **{
            f'dims_{k}': v for k, v in datasets.dims.items()
        },
        dims_emb=emb_dim,
        num_cls=datasets.num_classes,
        base_clf_cfg=json.dumps(base_clf_cfg),
        **{
            f'samples_{k}': v for k, v in common_cfg['samples'].items()
        },
    )

    metrics = []

    for dataset in tqdm(datasets, desc='Datasets', total=datasets.num_datasets):
        # Model path
        mp = (
            model_path
            .replace('${NAME}', model_cfg['name'])
            .replace('${IDX}', str(dataset.idx))
        )
        os.makedirs(os.path.dirname(mp), exist_ok=True)

        # Losses path
        lp = (
            losses_path
            .replace('${NAME}', model_cfg['name'])
            .replace('${IDX}', str(dataset.idx))
        )
        os.makedirs(os.path.dirname(lp), exist_ok=True)

        # Logs path
        logp = logs_path.replace('${NAME}', model_cfg['name'])

        # Callbacks
        logger = ptl_log.TensorBoardLogger(
            save_dir=logp,
            name='',
            version=str(dataset.idx),
        )

        es = ptl_cb.EarlyStopping(
            monitor='auc',
            patience=hparams.early_stopping * 2,
            mode='max',
            verbose=True,
            strict=True,
        )

        chpkt = ae_cb.CustomModelCheckpoint(
            filepath=mp,
            monitor='auc',
            save_top_k=1,
            mode='max',
            period=1,
            verbose=True,
        )

        loss_acc = ae_cb.LossAccumulator(filepath=lp)

        # Build model
        ae_model = ae_mod.UnsupervisedModel(hparams=hparams)

        # Create loaders
        tr_dl = dataset.get_loader(
            split='train',
            batch_size=hparams.batch_size,
            shuffle=True,
        )
        tr_dl_sorted = dataset.get_loader(
            split='train',
            batch_size=hparams.batch_size,
            shuffle=False,
        )
        val_dl = dataset.get_loader(
            split='val',
            batch_size=hparams.batch_size,
            shuffle=False,
        )

        # Initialize base classifier data
        ae_model.set_base_clf_data(
            train_data=tr_dl_sorted,
            val_data=val_dl,
        )

        # Train
        trainer = ptl.Trainer(
            logger=logger,
            checkpoint_callback=chpkt,
            early_stop_callback=es,
            callbacks=[loss_acc],
            max_epochs=num_epochs,
            progress_bar_refresh_rate=1,
            num_sanity_val_steps=0,
            gpus=1 if torch.cuda.is_available() else None,
            track_grad_norm=2,
        )
        trainer.fit(
            model=ae_model,
            train_dataloader=tr_dl,
            val_dataloaders=val_dl,
        )

        # Evaluate
        test_dl = dataset.get_loader(
            split='test',
            batch_size=hparams.batch_size,
            shuffle=False,
        )
        best_ae_model = ae_mod.UnsupervisedModel.load_from_checkpoint(
            checkpoint_path=list(chpkt.best_k_models.keys())[0]
        )
        best_ae_model.eval()
        best_ae_model.freeze()

        dss_metrics = {}

        for name, dl in (('train', tr_dl_sorted), ('val', val_dl), ('test', test_dl)):  # noqa
            dss_metrics[name] = best_ae_model.eval_on_base_clf(data=list(dl))

        metrics.append(dss_metrics)

        # Extract embedding vectors
        vecs = []
        for name, dl in (('train', tr_dl), ('val', val_dl), ('test', test_dl)):
            vecs.extend([
                v.numpy()
                for v in best_ae_model.data_to_emb_labels(data=list(dl))[0]
            ])

        vp = (
            vectors_path
            .replace('${NAME}', model_cfg['name'])
            .replace('${IDX}', str(dataset.idx))
        )
        os.makedirs(os.path.dirname(vp), exist_ok=True)

        with open(vp, 'wb') as fout:
            pickle.dump(vecs, fout)

    # Save metrics
    mtrp = metrics_path.replace('${NAME}', model_cfg['name'])
    os.makedirs(os.path.dirname(mtrp), exist_ok=True)

    with open(mtrp, 'wb') as fout:
        pickle.dump(metrics, fout)


if __name__ == '__main__':
    main()
