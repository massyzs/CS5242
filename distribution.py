import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from skimage.transform import resize
train_path = "/home/xiao/code/CS5242/dataset_copy/train"
test_path = "/home/xiao/code/CS5242/dataset_copy/test"
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.decomposition import PCA



def load_images_from_path(path):
    images = []
    labels = []
    for class_label, class_dir in enumerate(os.listdir(path)):
        class_path = os.path.join(path, class_dir)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = imread(image_path)
            images.append(image)
            labels.append(class_label)
    return images, np.array(labels)

train_images, train_labels = load_images_from_path(train_path)
test_images, test_labels = load_images_from_path(test_path)

def preprocess_images(images, target_size=(64, 64)):
    resized_images = [cv2.resize(image, target_size, interpolation=cv2.INTER_AREA) for image in images]
    gray_images = [rgb2gray(image) for image in resized_images]
    flattened_images = [image.flatten() for image in gray_images]
    return np.array(flattened_images)

train_images_processed = preprocess_images(train_images)
test_images_processed = preprocess_images(test_images)

reducer = PCA(n_components=2)
reducer.fit(train_images_processed)

train_embedded = reducer.transform(train_images_processed)
test_embedded = reducer.transform(test_images_processed)

def plot_distribution(embedded_data, labels, title, ax):
    for label in np.unique(labels):
        indices = np.where(labels == label)
        ax.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'Class {label}', alpha=0.6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

plot_distribution(train_embedded, train_labels, "Train Set Feature Distribution", ax1)
ax1.set_title("Train Set Feature Distribution")
ax1.legend()

plot_distribution(test_embedded, test_labels, "Test Set Feature Distribution", ax2)
ax2.set_title("Test Set Feature Distribution")
ax2.legend()
plt.savefig('/home/xiao/code/CS5242/CS5242/feature_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
