import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import re
import os
import bert

from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights

SAVED_MODEL_PATH="review_classifier/sentiment_bert_model.h5"
vocab_path="review_classifier/vocab.txt"
MAX_LENGTH=256


class _Review_Classifier_Service():
    _sentiment_bert_model=None
    TAG_RE = re.compile(r'<[^>]+>')
    _isinstance=None

    def remove_tags(self,text):
        return self.TAG_RE.sub('', text)  

    def preprocess_text(self,sen):

        #Removing HTML tags
        sentence=self.remove_tags(sen)

        #Select only alphabet characters
        sentence=re.sub('[^A-Za-z]',' ',sentence)

        #Removing multiple spaces
        sentence=re.sub(r"\s+"," ", sentence)

        #Remove the single character
        #sentence=re.sub(r"\s+[A-Za-z]+\s+",' ', sentence)

        return sentence

    def tokenize_reviews(self,reviews):
        print("in tokenize_review_function",reviews)
        tokenizer=bert.bert_tokenization.FullTokenizer(vocab_file=vocab_path,do_lower_case=True)  
        tokenized_reviews=[]
        for review in reviews:
            tokens=tokenizer.tokenize(review)
            if len(tokens)>MAX_LENGTH-2:
                tokens=tokens[:MAX_LENGTH-2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            token_ids=tokenizer.convert_tokens_to_ids(tokens)
            tokenized_reviews.append(token_ids)
        return tokenized_reviews

    def padding_tokenize_reviews(self,tokenized_reviews):
        tokens_with_paading=tf.keras.preprocessing.sequence.pad_sequences(tokenized_reviews,maxlen=MAX_LENGTH,padding="post",truncating="post",value=0)
        tokens_with_paading=np.array(tokens_with_paading)
        return tokens_with_paading

    def customizing_input(self,input_text):
        input_tokens=[]
        raw_text_input=input_text
        process_input_text=self.preprocess_text(raw_text_input)
        #print("process_input_text",process_input_text)
        input_tokens.append(process_input_text)
        #print("input_tokens", input_tokens)
        tokenize_input_text=self.tokenize_reviews(input_tokens)
        #print("tokenize_input_text",tokenize_input_text)
        tokenize_input_text_with_padding=self.padding_tokenize_reviews(tokenize_input_text)
        #print("tokenize_input_text_with_padding",tokenize_input_text_with_padding)
        return tokenize_input_text_with_padding

    def predict_review(self,input_text):
        model_input=self.customizing_input(input_text)
        prediction=self._sentiment_bert_model.predict(model_input)
        print("prediction is",prediction)
        return prediction

def Review_Classifier_Service():
    if _Review_Classifier_Service._isinstance is None:
        _Review_Classifier_Service._isinstance=_Review_Classifier_Service()
        _Review_Classifier_Service._sentiment_bert_model=keras.models.load_model(SAVED_MODEL_PATH,custom_objects={"BertModelLayer": bert.model.BertModelLayer})
    return _Review_Classifier_Service._isinstance

if __name__=='__main__':
    input_text="Hey my_name is prakash. I am not just doing a NLP intro.Very Very great. Excellent"
    rcs=Review_Classifier_Service()
    prediction=rcs.predict_review(input_text)
