from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpRequest
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import urllib, base64
import requests

IS_CALLED=False


def predict(request):
    if request.method== "POST":
        audio_file=request.FILES  
        response=requests.post("http://127.0.0.1:5000/predict_cnn",files=audio_file)
        output_response=response.json()
        output_img=create_plot(output_response)
        return render(request,"music_genre_classifier/predict_cnn.html",{"output_image":output_img})    
    return render(request,"music_genre_classifier/predict_cnn.html")
    
def create_plot(output_response):
    prediction=output_response["predictions"]
    genres=[
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
    plt.clf()
    x_pos=np.arange(len(genres))
    plt.barh(x_pos,prediction,label="genres")
    plt.yticks(x_pos,genres)
    plt.xlabel("Prediction")
    plt.title("Music Genre Classifier using CNN")
    plt.legend()
    fig = plt.gcf()
    #convert graph into dtring buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri =  urllib.parse.quote(string)
    IS_CALLED=True
    return uri

def home(request):
    return render(request,"music_genre_classifier/home.html")
