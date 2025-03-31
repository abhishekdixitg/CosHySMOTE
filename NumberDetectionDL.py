import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam
from tf_keras.layers import Dense, Dropout
from tf_keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# scale
X_train = X_train/255
X_test = X_test/255

X_train_flatten = X_train.reshape(len(X_train), 28*28)
X_test_flatten = X_test.reshape(len(X_test), 28*28)


model = Sequential([
    Dense(100, activation= 'relu', input_shape = (784,)),
    Dense(10, activation= 'sigmoid'),
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_flatten, y_train, epochs=10, batch_size=32, validation_data = ())

y_pred = model.predict(X_test_flatten)
print(np.argmax(y_pred[0]))

