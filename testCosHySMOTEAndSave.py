import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import itertools

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tf_keras import backend as K
from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from loadData import readData
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
from COSHysmote_V5 import COSHySMOTE
import kagglehub
#kmader_skin_cancer_mnist_ham10000_path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
#print(kmader_skin_cancer_mnist_ham10000_path)


with open("original_data.pkl", "rb") as f1:
    x_train, x_validate,x_test, y_train, y_validate = pickle.load(f1)

pca = PCA(n_components=500)  # Retain 500 principal components
x_train = pca.fit_transform(x_train)

# Calculate class counts
class_counts = Counter(np.argmax(y_train, axis=1))  # Use np.argmax if y_train is one-hot encoded
classes = list(class_counts.keys())  # Get all unique classes

# Determine total samples for balancing
total_samples = max(class_counts.values())  # Use the maximum count as the target for oversampling

# Desired samples for each class (make all classes proportional)
target_distribution = {cls: total_samples for cls in classes}

# Adjust retained samples and cluster sizes for each class
retained_samples = {cls: max(1, min(100, class_counts[cls] // 1000)) for cls in classes}
cluster_sizes = {cls: retained_samples[cls] for cls in classes}

# COSHySMOTE resampling
coshysmote = COSHySMOTE(
    target_distribution=target_distribution,
    cluster_sizes=cluster_sizes,
    random_state=42
)
X_resampled, y_resampled = coshysmote.fit_resample(x_train, np.argmax(y_train, axis=1))  # Convert one-hot to integer labels

# Check new resampled distribution
resampled_class_counts = Counter(y_resampled)
print("Resampled class distribution:", dict(resampled_class_counts))

# Visualize the resampled class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_resampled, palette="tab10")  # Use tab10 for more class colors
plt.title("Resampled Class Distribution After COSHySMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
# Save using pickle
with open("resampled_data.pkl", "wb") as f:
    pickle.dump((X_resampled, y_resampled), f)