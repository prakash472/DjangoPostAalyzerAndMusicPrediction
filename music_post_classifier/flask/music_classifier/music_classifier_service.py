import tensorflow.keras as keras
import numpy as np
import librosa
import matplotlib.pyplot as plt

SAVED_MODEL_PATH="music_classifier/cnn_model.h5"
SAMPLES_TO_CONSIDER = 66150



class _Music_Classifier_Service():

    _cnn_model=None
    _mappings=[ 
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]
    _isinstance=None

    def predict(self,file_path):
        #Extracting MFCC of audio file
        MFCCs=self.calculate_mfccs(file_path)

        # Convert 2d MFCC to 4d for input of CNN
        MFCCs=MFCCs[np.newaxis, ..., np.newaxis]

        #Making predictions
        prediction=self._cnn_model.predict(MFCCs)
        predicted_index=np.argmax(prediction)
        predicted_genre=self._mappings[predicted_index]
        print("prediction is",predicted_genre)
        return prediction[0]

    def calculate_mfccs(self,file_path,n_mfcc=13,n_fft=2048,hop_length=512):
        #load file
        signal,sr=librosa.load(file_path)

        #making sure the test audio is suited for our training
        if len(signal)>SAMPLES_TO_CONSIDER:
            signal=signal[:SAMPLES_TO_CONSIDER]
        
        #extract MFCCs
        MFCCs=librosa.feature.mfcc(signal,sr=sr,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
        MFCCs=MFCCs.T
        print("MFCCs shape is ",MFCCs.shape)
       
        return MFCCs



def Music_Classifier_Service():
    if _Music_Classifier_Service._isinstance is None:
        _Music_Classifier_Service._isinstance=_Music_Classifier_Service()
        _Music_Classifier_Service._cnn_model=keras.models.load_model(SAVED_MODEL_PATH)
    return _Music_Classifier_Service._isinstance

if __name__=='__main__':
    mcs=Music_Classifier_Service()  
    keyword_2=mcs.predict("test/pop.wav")

    plt.bar(mcs._mappings,keyword_2[0])
    plt.xlabel("genres")
    plt.ylabel("predictions")
    plt.show()
