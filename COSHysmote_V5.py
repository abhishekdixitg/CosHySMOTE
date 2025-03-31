import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class COSHySMOTE:
    def __init__(self, target_distribution=None, cluster_sizes=None, random_state=None):
        """
        Initialize COSHySMOTE.

        Parameters:
        - target_distribution: dict, desired class distribution after resampling. Example: {0: 10, 1: 20}.
        - cluster_sizes: dict, number of clusters to use per class. Example: {0: 5, 1: 3}.
        - random_state: int, random state for reproducibility.
        """
        self.target_distribution = target_distribution
        self.cluster_sizes = cluster_sizes
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Fit and resample the dataset.

        Parameters:
        - X: numpy array, feature matrix.
        - y: numpy array, labels.

        Returns:
        - X_resampled: numpy array, resampled feature matrix.
        - y_resampled: numpy array, resampled labels.
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        original_distribution = dict(zip(unique_classes, class_counts))

        print(f"Original class distribution: {original_distribution}")

        if self.target_distribution is None:
            raise ValueError("Target distribution must be specified.")

        X_resampled, y_resampled = [], []
        
        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            X_class = X[class_indices]
            n_clusters = self.cluster_sizes.get(cls, min(len(X_class), 5))  # Default to min(len, 5)

            # Clustering
            print(f"Clustering {len(X_class)} samples into {n_clusters} clusters for class {cls}.")
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X_class)
            
            # Retain one sample per cluster
            retained_samples = self._retain_samples(X_class, cluster_labels, kmeans)
            X_resampled.extend(retained_samples)
            y_resampled.extend([cls] * len(retained_samples))
            print(f"Clusters: {n_clusters}, Retained Samples for class {cls}: {len(retained_samples)}")

            # Generate synthetic samples if needed
            target_count = self.target_distribution.get(cls, len(retained_samples))
            if target_count > len(retained_samples):
                synthetic_count = target_count - len(retained_samples)
                synthetic_samples = self._generate_synthetic_samples(
                    X_class, retained_samples, cluster_labels, kmeans, synthetic_count
                )
                X_resampled.extend(synthetic_samples)
                y_resampled.extend([cls] * len(synthetic_samples))
                print(f"Generated {len(synthetic_samples)} synthetic samples for class {cls}.")

        # Verify final distribution
        resampled_distribution = dict(zip(*np.unique(y_resampled, return_counts=True)))
        print(f"Resampled class distribution: {resampled_distribution}")

        return np.array(X_resampled), np.array(y_resampled)

    def _retain_samples(self, X_class, cluster_labels, kmeans):
        """
        Retain one sample per cluster.

        Parameters:
        - X_class: numpy array, samples of the class.
        - cluster_labels: numpy array, cluster assignments.
        - kmeans: fitted KMeans object.

        Returns:
        - List of retained samples.
        """
        retained_samples = []
        for cluster_id in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_center = kmeans.cluster_centers_[cluster_id]
            closest_idx = cluster_indices[
                np.argmin(np.linalg.norm(X_class[cluster_indices] - cluster_center, axis=1))
            ]
            retained_samples.append(X_class[closest_idx])
        return retained_samples

    def _generate_synthetic_samples(self, X_class, retained_samples, cluster_labels, kmeans, synthetic_count):
        """
        Generate synthetic samples.

        Parameters:
        - X_class: numpy array, samples of the class.
        - retained_samples: list, retained samples.
        - cluster_labels: numpy array, cluster assignments.
        - kmeans: fitted KMeans object.
        - synthetic_count: int, number of synthetic samples to generate.

        Returns:
        - List of synthetic samples.
        """
        synthetic_samples = []
        nearest_neighbors = NearestNeighbors(n_neighbors=5).fit(X_class)

        for _ in range(synthetic_count):
            cluster_id = np.random.choice(np.unique(cluster_labels))
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            random_idx = np.random.choice(cluster_indices)
            base_sample = X_class[random_idx]

            # Find neighbors within the cluster
            neighbors = nearest_neighbors.kneighbors(base_sample.reshape(1, -1), return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbors)
            neighbor_sample = X_class[neighbor_idx]

            # Interpolate to create a synthetic sample
            alpha = np.random.uniform(0, 1)
            synthetic_sample = base_sample + alpha * (neighbor_sample - base_sample)
            synthetic_samples.append(synthetic_sample)

        return synthetic_samples