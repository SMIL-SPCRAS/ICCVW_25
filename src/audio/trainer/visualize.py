import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def conf_matrix(targets, predicts, labels):
    """
    Compute confusion matrix.
    """
    return confusion_matrix(targets, predicts, labels=labels)


def plot_conf_matrix(cm, labels, title, save_path=None, normalize=False, figsize=(8, 6), xticks_rotation=45):
    """
    Plot and optionally save a confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix.
        labels (list): List of class names.
        title (str): Title of the plot.
        save_path (str): File path to save the plot (e.g., .png, .svg).
        normalize (bool): Whether to normalize the confusion matrix.
        figsize (tuple): Size of the figure.
        xticks_rotation (int): Rotation angle for x-axis tick labels.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    counts = cm.astype(int)
    percentages = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100
    annot = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{counts[i, j]}\n{percentages[i, j]:.2f}%"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(percentages, annot=annot, fmt="s", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, )

    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=xticks_rotation)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig