import librosa
import numpy as np
import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
DATA_PATH = "music_data.json"

def load_data(data_path):
   
    with open(data_path, "r") as fp:
        data = json.load(fp)
    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["label"])

    print("Data succesfully loaded!")

    return  X, y

def create_data(test_size,validation_size):
    X,y=load_data(DATA_PATH)

    #create training and test data
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=test_size)

    #create training and validation data
    X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=validation_size)

   

    return X_train,X_test,X_validate,y_train,y_test,y_validate
    

def create_model(input_shape):
    model=keras.Sequential()
    #Input Layer
    model.add(keras.layers.Input(shape=(input_shape)))
   
    # 1st LSTM layer
    model.add(keras.layers.LSTM(64,input_shape=input_shape,activation="tanh",return_sequences=True))

    # 2nd LSTM layer
    model.add(keras.layers.LSTM(64))

    #Add a dense layers
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10,activation="softmax"))

    return model


if __name__ == "__main__":
    X_train,X_test,X_validate,y_train,y_test,y_validate=create_data(0.25,0.2)

    # building a RNN model
    input_shape=(X_train.shape[1],X_train.shape[2])
    model=create_model(input_shape)

    #compiling the model
    optimiser=keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimiser, loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    #training the model
    model.fit(X_train,y_train,validation_data=(X_validate,y_validate),batch_size=32,epochs=30)

    #evaluate the model
    test_error, test_accuracy= model.evaluate(X_test,y_test,verbose=1)
    print("The accuracy on the test data is {}".format(test_accuracy))




    