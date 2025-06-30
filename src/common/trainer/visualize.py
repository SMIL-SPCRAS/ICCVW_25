import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

import torch


def plot_conf_matrix(cm: np.ndarray, labels: list[str], 
                     title: str, save_path: str = None, 
                     normalize: bool = False, 
                     figsize: tuple[int, int] = (8, 6), 
                     xticks_rotation: int = 45) -> Figure:
    """
    Plot and optionally save a confusion matrix.
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


def visualize_uncertainty(logvars: torch.Tensor, emotion_labels: list[str] = None, save_path: str = None) -> None:
    """
    Visualize log-variance (uncertainty) as a heatmap for emotion predictions using seaborn.
    """
    logvars = logvars.detach().cpu().numpy()
    batch_size, num_emotions = logvars.shape

    plt.figure(figsize=(num_emotions, batch_size * 0.5))
    ax = sns.heatmap(logvars, cmap="viridis", cbar=True, xticklabels=emotion_labels, yticklabels=False)
    ax.set_title("Emotion Prediction Uncertainty (logvar)")
    ax.set_xlabel("Emotions")
    ax.set_ylabel("Samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    
    plt.close()
