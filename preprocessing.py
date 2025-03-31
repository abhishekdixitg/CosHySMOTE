import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import itertools

from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tf_keras import backend as K
from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from loadData import readData
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle
from COSHysmote_V5 import COSHySMOTE

def preprocessing(path) :
    features, target = readData(path)

    x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
    x_train = np.asarray(x_train_o['image'].tolist())
    x_test = np.asarray(x_test_o['image'].tolist())

    x_train_mean = np.mean(x_train)
    x_train_std = np.std(x_train)

    x_test_mean = np.mean(x_test)
    x_test_std = np.std(x_test)

    x_train = (x_train - x_train_mean)/x_train_std
    x_test = (x_test - x_test_mean)/x_test_std
    y_train = to_categorical(y_train_o, num_classes = 7)
    y_test = to_categorical(y_test_o, num_classes = 7)
    print("Shape of y_train:", y_train.shape)

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)

    print("Shape of y_train:", y_train.shape)

    # Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
    x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
    x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
    #x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    #x_validate = x_validate.reshape(x_validate.shape[0], -1)
    print(f"Flattened X_class shape: {x_train.shape}")
    print("Shape of y_train:", y_train.shape)
    # Save using pickle
    with open("original_data.pkl", "wb") as f1:
        pickle.dump((x_train, x_validate,x_test, y_train, y_validate), f1)

if __name__ == "__main__":
    preprocessing()