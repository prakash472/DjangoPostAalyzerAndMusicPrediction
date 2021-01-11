import librosa
import numpy as np
import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
DATA_PATH = "music_data.json"
SAVED_MODEL_PATH="../flask/music_classifier/cnn_model.h5"

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

    """
    Here in conv neural network, the input is 3d. But our training data consists of (no_of_samples,130,13) where 130 is the 
    frames or frequencies where each of them has 13 mfccs. Now we need to convert it into 3d instead of 2d.
    In 3d (usually for image, 3rd dimension is for depth i.e if RGB it is (X,Y,3) if black and white it is 
    (X,Y,1). So, we change our (130,13) to (130,13,1) since audio has no depth. (No_samples dimesion is not included for input))
    """
    X_train=X_train[..., np.newaxis]  # Old_X_train.shape=(5991, 130, 13), #New_X_train_shape=(5991, 130, 13, 1), Actually 4d
    X_test=X_test[..., np.newaxis]
    X_validate=X_validate[..., np.newaxis]
    y_train=y_train[..., np.newaxis] 
    y_test=y_test[..., np.newaxis]
    y_validate=y_validate[..., np.newaxis]

    return X_train,X_test,X_validate,y_train,y_test,y_validate
    

def create_model(input_shape):
    model=keras.Sequential()
    #Input Layer
    model.add(keras.layers.Input(shape=(input_shape)))
    # 1st conv layer 
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same"))
    model.add(keras.layers.BatchNormalization())

    #2nd conv layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3),activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same"))
    model.add(keras.layers.BatchNormalization())

    #3rd conv layer 
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(2,2),activation="relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="same"))
    model.add(keras.layers.BatchNormalization())

    #Flattening the input and feeding it to the dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,activation="relu"))
    model.add(keras.layers.Dropout(0.2))

    #output layer
    model.add(keras.layers.Dense(10,activation="softmax"))

    return model


if __name__ == "__main__":
    X_train,X_test,X_validate,y_train,y_test,y_validate=create_data(0.25,0.2)

    # building a CNN model
    input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])
    model=create_model(input_shape)

    #compiling the model
    optimiser=keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimiser, loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    #training the model
    model.fit(X_train,y_train,validation_data=(X_validate,y_validate),batch_size=32,epochs=30)

    #evaluate the model
    test_error, test_accuracy= model.evaluate(X_test,y_test,verbose=1)
    print("The accuracy on the test data is {}".format(test_accuracy))

    #save the model
    model.save(SAVED_MODEL_PATH)





    