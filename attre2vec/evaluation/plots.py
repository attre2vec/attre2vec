"""Code for making basic paper plots."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_auc_dimensionality(
    frame_with_results: pd.DataFrame,
    key: str = "dataset",
    x_label: str = "Dataset",
) -> plt.Axes:
    """Plots AUC value given dimensionalities of methods.

    Args:
        frame_with_results: DataFrame where each record holds what should be
            visualized on the X axis, "auc" key, and "method" key.
        key: string of what should be visualized on X axis.
        x_label: string for the label on the X axis.
    Returns:
        Single axis of matplotlib with visualized data.

    """
    fig, ax = plt.subplots()
    sns.set_context("paper")
    sns.set_palette("cubehelix")
    frame_with_results = frame_with_results.rename(
        {key: x_label, "auc": "AUC"}, axis=1
    )

    ax = sns.barplot(
        x_label, "AUC", hue="method", ax=ax, data=frame_with_results
    )
    ax.get_legend().set_title("")
    ax.grid(True, linestyle=":")

    return ax


def _test_plotting():
    import numpy as np

    data = []
    rng = np.random.RandomState(seed=0)
    for dims in [16, 32, 64, 256]:
        for method in ["sage", "line2vec", "nodegate", "attr2vec"]:
            for auc in rng.uniform(0.5, 1.0, size=(10,)):
                data.append({"dim": dims, "method": method, "auc": auc})

    frame = pd.DataFrame(data=data)
    plot_auc_dimensionality(frame, key="dim")
    plt.show()


if __name__ == "__main__":
    _test_plotting()
