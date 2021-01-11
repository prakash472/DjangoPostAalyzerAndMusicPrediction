import librosa
import numpy as np
import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
DATA_PATH = "music_data.json"

def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)
    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["label"])

    print("Data succesfully loaded!")

    return  X, y

def plt_history(history):
    #plot accuracy
    fig, axes=plt.subplots(2,1)
    axes[0].plot(history.history["accuracy"])
    axes[0].plot(history.history["val_accuracy"])
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(["train","test"],loc="upper left")

    #plot error or loss
    axes[1].plot(history.history["loss"])
    axes[1].plot(history.history["val_loss"])
    axes[1].set_title("Model Error")
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend(["train","test"],loc="upper left")
    
    plt.show()

if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    #Creating train and test data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    #Build model
    model=keras.Sequential()
    #input layer
    model.add(keras.layers.Flatten(input_shape=(X.shape[1],X.shape[2])))
    #1st dense layer
    model.add(keras.layers.Dense(512,activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)))
    model.add(keras.layers.Dropout(0.2))
    #2nd dense layer
    model.add(keras.layers.Dense(256,activation="relu", kernel_regularizer=keras.regularizers.l2(0.002)))
    model.add(keras.layers.Dropout(0.2))
    #3rd dense layer 
    model.add(keras.layers.Dense(64,activation="relu",  kernel_regularizer=keras.regularizers.l2(0.002)))
    model.add(keras.layers.Dropout(0.2))
    #output layer
    model.add(keras.layers.Dense(10,activation="softmax"))

    #compile model
    optimiser=keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimiser,loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    model.summary()

    #train_model
    history=model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=32,epochs=62) 

    #plot accuracy and loss
    plt_history(history) 
    