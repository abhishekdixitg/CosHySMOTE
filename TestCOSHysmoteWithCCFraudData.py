import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from COSHysmote_V3 import COSHySMOTE
from itertools import combinations
from sklearn.utils import class_weight
import seaborn as sns
from collections import Counter
# Confusion matrix
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from imblearn.datasets import make_imbalance
from sklearn.datasets import load_iris
# Load the dataset
import kagglehub
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold


colors = ["#0101DF", "#DF0101"]
# Download latest version
#path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
path = 'C:/Users/abhishekd/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3'
print("Path to dataset files:", path)
df = pd.read_csv(path+'/creditcard.csv')

# Split features and labels
X = df.drop(columns=["Class"])
y = df["Class"]

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Check class distribution
print("Original class distribution:", dict(pd.Series(y_train).value_counts()))


# Adjust target samples for balanced distribution
total_majority = len(y_train[y_train == 0])
total_minority = len(y_train[y_train == 1])

# Desired total samples for each class
desired_minority_samples = total_majority // 2  # Keep the minority class size proportional to the majority
desired_majority_samples = desired_minority_samples

# Adjust retained samples and target synthetic samples
retained_samples = {
    0: min(100, total_majority // 1000),
    1: min(50, total_minority // 10)
}

target_samples = {
    0: desired_majority_samples,
    1: desired_minority_samples
}

# Apply COSHySMOTE
coshysmote = COSHySMOTE(
    target_samples,
    retained_samples,
    random_state=42
)
X_resampled, y_resampled = coshysmote.fit_resample(X_train, y_train)

# Check new resampled distribution
print("Resampled class distribution:", dict(Counter(y_resampled)))

# Visualize the resampled class distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_resampled, palette=["#0101DF", "#DF0101"])
plt.title("Resampled Class Distribution After COSHySMOTE")
plt.show()

