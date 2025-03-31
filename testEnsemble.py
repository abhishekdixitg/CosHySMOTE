from imblearn.ensemble import EasyEnsembleClassifier
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

with open("resampled_data.pkl", "rb") as f:
    X_resampled_loaded, y_resampled_loaded = pickle.load(f)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled_loaded, y_resampled_loaded, test_size=0.2, random_state=42)

# Initialize EasyEnsemble classifier
easy_ensemble = EasyEnsembleClassifier(random_state=42, n_estimators=10)
easy_ensemble.fit(X_train, y_train)

# Evaluate
y_pred_easy = easy_ensemble.predict(X_test)
print(classification_report(y_test, y_pred_easy))
