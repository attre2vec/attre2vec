"""Utility functions."""
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics as sk_mtr
import torch


def calc_metrics(y_score, y_pred, y_true, max_cls):
    """Calculates metrics."""
    metrics = {
        **sk_mtr.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
        ),
        'auc': sk_mtr.roc_auc_score(
            y_true=y_true,
            y_score=y_score,
            average='weighted',
            multi_class='ovr',
            labels=range(max_cls),
        ),
        'cm': sk_mtr.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=range(max_cls),
        )
    }

    return metrics


def mk_stack(
    h_xe: Dict[Tuple[int, int], Union[List[float], torch.Tensor]],
    xe: torch.Tensor,
) -> torch.Tensor:
    """Creates stacked tensor based on current mapping and edge list."""
    return torch.stack([
        mk_tensor(h_xe[tuple(edge)])
        for edge in xe.tolist()
    ], dim=0)


def mk_tensor(t: Union[List[float], torch.Tensor]) -> torch.Tensor:
    """Converts to tensor if is list, else does nothing."""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float)
    return t
