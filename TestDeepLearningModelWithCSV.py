from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam

df_train = pd.read_csv("./data/emotion_dataset/train.txt",delimiter=';', header=None, names=['sentence','label'])
#print(df_train['label'].unique())
df_test = pd.read_csv("./data/emotion_dataset/test.txt",delimiter=';', header=None, names=['sentence','label'])
df_val = pd.read_csv("./data/emotion_dataset/val.txt",delimiter=';', header=None, names=['sentence','label'])

df_train = df_train[~df_train['label'].str.contains('love')]
df_train = df_train[~df_train['label'].str.contains('surprise')]

df_test = df_test[~df_test['label'].str.contains('love')]
df_test = df_test[~df_test['label'].str.contains('surprise')]

df_val = df_val[~df_val['label'].str.contains('love')]
df_val = df_val[~df_val['label'].str.contains('surprise')]


tr_text = df_train['sentence']
tr_label = df_train['label']

ts_text = df_test['sentence']
ts_label = df_test['label']

vl_text = df_val['sentence']
vl_label = df_val['label']

# Step 1: Text to Numerical Features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train = tfidf_vectorizer.fit_transform(tr_text).toarray()
X_test = tfidf_vectorizer.transform(ts_text).toarray()
X_val = tfidf_vectorizer.transform(vl_text).toarray()

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(tr_label)
y_test = label_encoder.fit_transform(ts_label)
y_val = label_encoder.fit_transform(vl_label)



model = Sequential([
    Dense(128, activation= 'relu', input_shape = (X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
y_pred = model.predict(X_test).flatten()

# Step 6: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

# Step 6: Visualize Results (Optional)
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.title("Linear Regression: True vs Predicted")
plt.show()