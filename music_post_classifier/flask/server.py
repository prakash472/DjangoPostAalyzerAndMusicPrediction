from flask import Flask,request,jsonify
import random
from music_classifier.music_classifier_service import Music_Classifier_Service
from review_classifier.review_classifier_service import Review_Classifier_Service
import os
import numpy as np


app=Flask(__name__)

@app.route("/predict_cnn",methods=["POST"])
def predict_cnn():

    #get audio-file and save
    audio_file=request.files["file"]
    file_name=str(random.randint(0,10000))
    audio_file.save(file_name)

    #invoke Music classfier service
    mcs=Music_Classifier_Service()
    
    #make_prediction
    prediction=mcs.predict(file_name)
    prediction=prediction.tolist()
    print("Prediction is from server",prediction)

    #remove audio-file after predicting
    os.remove(file_name)

    #send back the predicted keyword in json format
    data={"predictions":prediction}

    return jsonify(data)


@app.route("/predict_review",methods=["POST"])
def predict_review():

    #get audio-file and save
    review_text_json=request.get_json()
    review_text=review_text_json["review"]

    #invoke Music classfier service
    rcs=Review_Classifier_Service()
    
    #make_prediction
    prediction=rcs.predict_review(review_text)
    #prediction=prediction.tolist()
    print("Prediction is from server",prediction)

    #remove audio-file after predicting
    #os.remove(file_name)
    prediction=prediction.tolist()[0]

    #send back the predicted keyword in json format
    data={"predictions":prediction}

    return jsonify(data)


if __name__=="__main__":
    app.run(debug=True)
      