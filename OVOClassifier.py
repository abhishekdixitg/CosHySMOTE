from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from itertools import combinations
from tf_keras.models import Sequential
from tf_keras.layers import Dense
import numpy as np
from COSHysmote_V2 import COSHySMOTE

def calculate_class_weights(y, num_classes=None):
    """
    Calculate class weights for binary classification in OVO setting.
    If num_classes is not provided, weights are computed for present classes.
    """
    unique_classes = np.unique(y)  # Use only present classes in y
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y
    )
    return {cls: weight for cls, weight in zip(unique_classes, class_weights)}


# Function to create and train OVO classifiers
def ovo_classifiers(X, y, num_classes, n_to_sample=50):
    classifiers = {}
    combs = list(combinations(range(num_classes), 2))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize the features
    
    for c1, c2 in combs:
        # Filter data for the two classes
        idx = np.where((y == c1) | (y == c2))[0]
        X_pair = X[idx]
        y_pair = y[idx]
        y_pair = np.where(y_pair == c1, 0, 1)  # Binary labels for the two classes

        # Apply COS-HYSMOTE for data augmentation
        cos_hysmote = COSHySMOTE(k_neighbors=5,sampling_factor=3, epsilon=0.1, delta=0.5)
        X_resampled, y_resampled = cos_hysmote.fit_resample(X_pair, y_pair)
        

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Create a simple neural network model
        model = Sequential([
            Dense(8, input_dim=X_train.shape[1], activation='relu'),
            Dense(4, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Compute class weights for balanced training
        #classes = np.unique(y_train)
    
        #weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
       # class_weight = calculate_class_weights(y_train) #dict(zip(classes, weights))
        class_weight = calculate_class_weights(y_train)
        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, class_weight=class_weight)
        
        # Store the classifier
        classifiers[(c1, c2)] = model
        
        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(f"Classifier for {c1} vs {c2}:")
        print(classification_report(y_test, y_pred))
    
    return classifiers
