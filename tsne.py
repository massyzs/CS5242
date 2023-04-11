import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
import pickle
# device=torch.device("cuda:0")
# Load a pre-trained model
# path="/home/xiao/code/CS5242/CS5242/tensor/*.dict"
# files=glob(path)
with open('/home/xiao/code/CS5242/CS5242/model/emb.pkl', 'rb') as f:
    tmp_dict = pickle.load(f)
print("checkpoint1")
embedding1_ls=tmp_dict["tensor1"]

embedding2_ls=tmp_dict["tensor2"]

labels=tmp_dict["labels"]
# embedding1 = torch.cat(embedding1).numpy()
# embedding2 = torch.cat(embedding2).numpy()
# labels = torch.cat(labels).numpy()
# Reduce the dimensionality using t-SNE
for i in range(len(embedding1_ls)):
    embedding1=embedding1_ls[i].reshape(512,-1)
    embedding2=embedding2_ls[i].reshape(512,-1)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_1 = tsne.fit_transform(embedding1)
    print("checkpoint2")
    embeddings_2 = tsne.fit_transform(embedding2)
    print("checkpoint3")
    # Visualize the 2D embeddings using seaborn

    fig,ax=plt.subplots(1,2)
    sns.scatterplot(x=embeddings_1[:, 0], y=embeddings_1[:, 1], palette='deep', legend='full', alpha=0.5,label="Before Norm",ax=ax[0])
    sns.scatterplot(x=embeddings_2[:, 0], y=embeddings_2[:, 1], palette='deep', legend='full', alpha=0.5,label="After Norm",ax=ax[1])
    fig.suptitle(f"T-SNE:{i}")
    plt.savefig(f"/home/xiao/code/CS5242/CS5242/model/tsne-{i}.jpg")
    if i==5:
        break