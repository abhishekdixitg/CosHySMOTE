import numpy as np
from sklearn.datasets import make_classification
from COSHysmote_V3 import COSHySMOTE
# Generate a large imbalanced dataset
X, y = make_classification(
    n_samples=10000,  # Total samples
    n_features=20,    # Number of features
    n_informative=15, # Number of informative features
    n_redundant=5,    # Number of redundant features
    weights=[0.9, 0.1],  # Class imbalance
    random_state=42
)

# Split majority and minority classes
X_majority = X[y == 0]
X_minority = X[y == 1]

# Define target distribution and cluster sizes
target_distribution = {0: 20, 1: 40}
cluster_sizes = {0: 5, 1: 3}

# Apply COSHySMOTE
coshysmote = COSHySMOTE(target_distribution=target_distribution, cluster_sizes=cluster_sizes, random_state=42)
X_resampled, y_resampled = coshysmote.fit_resample(X, y)

# Validate resampled distribution
from collections import Counter
print("Resampled class distribution:", Counter(y_resampled))
