import pandas as pd

import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("./data/spam/spam.csv", delimiter = ',', encoding='latin-1')

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace= True)

X = df.v2
y = df.v1

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(X).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

y = y.reshape(-1, 1)

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.15)

model = Sequential([
    Dense(128, activation= 'relu', input_shape = (X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer = Adam(learning_rate=0.001), loss='mse', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 10, batch_size = 32, validation_data = ())

# Evaluate the model
y_pred = model.predict(X_test).flatten()

# Step 6: Evaluate the Model
mse = mean_squared_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 6: Visualize Results (Optional)
import matplotlib.pyplot as plt

plt.scatter(Y_test, y_pred, alpha=0.5)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("Linear Regression: True vs Predicted")
plt.show()