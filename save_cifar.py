import numpy as np
from datasets import load_dataset

cifar10_dataset = load_dataset("cifar10")

trainset = cifar10_dataset["train"]
testset = cifar10_dataset["test"]
# Convert the images and labels into NumPy arrays
images = np.array([np.array(img).reshape(3, 32, 32) for img in trainset["img"]])
labels = np.array(trainset["label"])
breakpoint()

# Save the images and labels as NPY files
np.save('cifar10_train_images.npy', images)
np.save('cifar10_train_labels.npy', labels)
