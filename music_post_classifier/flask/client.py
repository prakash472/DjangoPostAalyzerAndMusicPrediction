import requests

# server url
URL = "http://127.0.0.1:5000/predict_review"


# audio file we'd like to send for predicting keyword
TEXT_DATA = "Hey my_name is prakash. I am not just doing a NLP intro.Very Very bad. Excellent"


if __name__ == "__main__":
    review_text={"review": TEXT_DATA}
    response = requests.post(URL, json=review_text)
    data = response.json()

    print("Predicted Review is: {}".format(data["predictions"]))