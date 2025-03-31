import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
import matplotlib.pyplot as plt
from OVOClassifier import ovo_classifiers

# Load the dataset from the UCI repository
repo_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
data = pd.read_csv(repo_url, delimiter=',', header=None)

# Print dataset shape
print("Dataset shape:", data.shape)

# Split into features (X) and labels (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Replace '?' with NaN
X = X.replace('?', np.NaN)

# Remove unwanted columns (more than 40% missing values)
thresh = len(X) * 0.4
X.dropna(thresh=thresh, axis=1, inplace=True)

# Impute missing values with the median
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='median')
X = pd.DataFrame(imp_mean.fit_transform(X))

# Normalize the features using StandardScaler
std_scaler = StandardScaler()
X = pd.DataFrame(std_scaler.fit_transform(X))

# Encode the class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Print the number of classes
num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

# Print class distribution
class_counts = Counter(y)
print("Class distribution:", class_counts)

# Plot class distribution
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.show()

# Train OVO classifiers
ovo_models = ovo_classifiers(X, y, num_classes)
