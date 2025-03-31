from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open("resampled_data.pkl", "rb") as f:
    X_resampled_loaded, y_resampled_loaded = pickle.load(f)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_loaded, y_resampled_loaded, test_size=0.2, random_state=42)

# Base model for Bagging and Boosting
base_model = RandomForestClassifier(random_state=42)

# Bagging OVA
bagging_ova = OneVsRestClassifier(BaggingClassifier(base_model, random_state=42))
bagging_ova.fit(X_train, y_train)
bagging_ova_predictions = bagging_ova.predict(X_test)
print("Bagging OVA Classification Report:")
print(classification_report(y_test, bagging_ova_predictions))
bagging_ova_accuracy = accuracy_score(y_test, bagging_ova_predictions)
print("Bagging OVA Accuracy:", bagging_ova_accuracy)