from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import numpy as np
from sklearn.decomposition import PCA

# Load data
with open("original_data.pkl", "rb") as f1:
    X_train, x_validate, X_test, y_train, y_test = pickle.load(f1)

# If y_train is one-hot encoded (shape: [n_samples, n_classes]), convert it to 1D labels
if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    y_train = np.argmax(y_train, axis=1)  # Convert one-hot encoding to class labels

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

pca = PCA(n_components=500)  # Retain 500 principal components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)  # Apply the same transformation to the test set


# Boosting OVA
boosting_ova = OneVsRestClassifier(XGBClassifier(random_state=42))
boosting_ova.fit(X_train, y_train)
boosting_ova_predictions = boosting_ova.predict(X_test)
print("Boosting OVA Classification Report:")
print(classification_report(y_test, boosting_ova_predictions))
boosting_ova_accuracy = accuracy_score(y_test, boosting_ova_predictions)
print("Boosting OVA Accuracy:", boosting_ova_accuracy)

