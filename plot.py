from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import scipy
import torch
import os
from sklearn.manifold import TSNE

# Load saved data
# plot linear curve
acc_list = []
plt.plot(range(len(acc_list)), acc_list, label = "test accuracy")
# plt.plot(range(len(self_mem_list)), [infl for infl in self_mem_list], label = "self-influence")
plt.title(r'Comparison of $g_z H_{\theta}^{-1} g_x$ and self-influence $$g_x H_{\theta}^{-1} g_x$')
plt.xlabel('iter')
plt.ylabel("influence function value")
plt.show()

#t-sne
class_names = ["x2","x1"]

# Tool functions

## T-sne
def plot_embedding(
    tsne_result, label, title, xlabel="tsne_x", ylabel="tsne_y", custom_palette=None, size=(10, 10)
):
    """Plot embedding for T-SNE with labels"""
    # Data Preprocessing
    if torch.is_tensor(tsne_result):
        tsne_result = tsne_result.cpu().numpy()
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    x_min, x_max = np.min(tsne_result, 0), np.max(tsne_result, 0)
    tsne_result = (tsne_result - x_min) / (x_max - x_min)

    # Plot
    tsne_result_df = pd.DataFrame(
        {"tsne_x": tsne_result[:, 0],
            "tsne_y": tsne_result[:, 1], "label": label}
    )
    fig, ax = plt.subplots(1, figsize=size)

    num_class = len(pd.unique(tsne_result_df["label"]))
    if custom_palette is None:
        custom_palette = sns.color_palette("hls", num_class)

    # s: maker size, palette: colors

    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue="label",
        data=tsne_result_df,
        ax=ax,
        s=40,
        palette=custom_palette,
        alpha=1,
    )
    #     sns.lmplot(x='tsne_x', y='tsne_y', hue='label',
    #                     data=tsne_result_df, size=9, scatter_kws={"s":20,"alpha":0.3},fit_reg=False, legend=True,)

    # Set Figure Style
    lim = (-0.01, 1.01)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #ax.tick_params(axis="x", labelsize=20)
    #ax.tick_params(axis="y", labelsize=20)
    ax.set_title(title)
    ax.set_aspect("equal")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    return fig

def get_embedding(data):
    """Get T-SNE embeddings"""
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    tsne = TSNE(n_components=2, init="pca", random_state=0)
    result = tsne.fit_transform(data)
    return result


def tsne_fig(
    data,
    label,
    title="t-SNE embedding",
    xlabel="tsne_x",
    ylabel="tsne_y",
    custom_palette=None,
    size=(10, 10)
):
    """Get T-SNE embeddings figure"""
    tsne_result = get_embedding(data)
    fig = plot_embedding(tsne_result, label, title, xlabel,
                         ylabel, custom_palette, size)
    return fig

def tsne(features, labels, log_dir, filename_contents):
    if not isinstance(features,np.ndarray):
        features = np.asarray(features)
    if not isinstance(labels,np.ndarray):
        labels = np.asarray(labels)
    num_of_models = filename_contents[0]
    cond_x = filename_contents[1]
    layer = filename_contents[2]

    print("features: ",type(features),type(labels))

    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    features = features[sort_idx]
    classes = class_names
    label_class = [classes[i].capitalize() for i in labels]

    # Plot T-SNE
    custom_palette = sns.color_palette("hls", len(np.unique(labels)))
    fig = tsne_fig(
        features,
        label_class,
        title=f"t-SNE Embedding of {layer}",
        xlabel="Dim 1",
        ylabel="Dim 2",
        custom_palette=custom_palette,
        size = (10,10)
    )
    plt.savefig(f"{log_dir}/lossFigs_{num_of_models}/{layer}_visualization_on_first_Ref-x1={cond_x}.png")