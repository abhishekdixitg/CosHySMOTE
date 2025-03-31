from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE
import pickle
from imblearn.combine import SMOTETomek

# Load data
with open("original_data.pkl", "rb") as f1:
    X_train, x_validate, X_test, y_train, y_test = pickle.load(f1)

# Apply ADASYN
bsmote = BorderlineSMOTE(random_state=42, kind="borderline-1")  # kind="borderline-1" or "borderline-2"
X_resampled, y_resampled = bsmote.fit_resample(X_train, y_train)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Apply PCA
pca = PCA(n_components=500)  # Retain 500 principal components
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)  # Apply the same transformation to the test set

# Base model for Bagging
base_model = RandomForestClassifier(random_state=42)

# Bagging OVA
bagging_ova = OneVsRestClassifier(BaggingClassifier(base_model, random_state=42))
bagging_ova.fit(X_train, y_train)
bagging_ova_predictions = bagging_ova.predict(X_test)

# Evaluate
print("Bagging OVA Classification Report with PCA Data:")
print(classification_report(y_test, bagging_ova_predictions, digits=4))
print("Bagging OVA Accuracy with PCA Data:", round(accuracy_score(y_test, bagging_ova_predictions),4))
