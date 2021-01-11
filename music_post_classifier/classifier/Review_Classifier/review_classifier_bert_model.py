import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import bert
import re
import os
import numpy as np

from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights

# Now let's tokenize every sentence from our reviews
SAVED_MODEL_PATH="sentiment_bert_model.h5"
MAX_LENGTH=256

bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir=os.path.join("model",bert_model_name)
bert_ckpt_file=os.path.join(bert_ckpt_dir,"bert_model.ckpt")
bert_config_file=os.path.join(bert_ckpt_dir,"bert_config.json")
MAX_LENGTH=256

TAG_RE = re.compile(r'<[^>]+>')
def create_model(max_seq_len, bert_ckpt_file,bert_config_file):
    print("max_len is",max_seq_len)
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read())
        bert_params  = stock_params.to_bert_model_layer_params()
        bert = BertModelLayer.from_params(bert_params, name="bert")
            
    input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
    bert_output = bert(input_ids)

    print("bert shape", bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=1, activation="sigmoid")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)   
    return model

def remove_tags(text):
    return TAG_RE.sub('', text)  

def preprocess_text(sen):

    #Removing HTML tags
    sentence=remove_tags(sen)

    #Select only alphabet characters
    sentence=re.sub('[^A-Za-z]',' ',sentence)

    #Removing multiple spaces
    sentence=re.sub(r"\s+"," ", sentence)

    #Remove the single character
    #sentence=re.sub(r"\s+[A-Za-z]+\s+",' ', sentence)

    return sentence

def tokenize_reviews(reviews):
    print("in tokenize_review_function",reviews)
    tokenizer=bert.bert_tokenization.FullTokenizer(vocab_file="vocab.txt",do_lower_case=True)  
    tokenized_reviews=[]
    for review in reviews:
        tokens=tokenizer.tokenize(review)
        if len(tokens)>MAX_LENGTH-2:
            tokens=tokens[:MAX_LENGTH-2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids=tokenizer.convert_tokens_to_ids(tokens)
        tokenized_reviews.append(token_ids)
    return tokenized_reviews

def padding_tokenize_reviews(tokenized_reviews):
    tokens_with_paading=tf.keras.preprocessing.sequence.pad_sequences(tokenized_reviews,maxlen=MAX_LENGTH,padding="post",truncating="post",value=0)
    tokens_with_paading=np.array(tokens_with_paading)
    return tokens_with_paading

def customizing_input(input_text):
    input_tokens=[]
    raw_text_input=input_text
    process_input_text=preprocess_text(raw_text_input)
    #print("process_input_text",process_input_text)
    input_tokens.append(process_input_text)
    #print("input_tokens", input_tokens)
    tokenize_input_text=tokenize_reviews(input_tokens)
    #print("tokenize_input_text",tokenize_input_text)
    tokenize_input_text_with_padding=padding_tokenize_reviews(tokenize_input_text)
    #print("tokenize_input_text_with_padding",tokenize_input_text_with_padding)
    return tokenize_input_text_with_padding


"""
def create_model(input_text):
        bert_model_name="uncased_L-12_H-768_A-12"
        bert_ckpt_dir=os.path.join("model",bert_model_name)
        bert_ckpt_file=os.path.join(bert_ckpt_dir,"bert_model.ckpt")
        bert_config_file=os.path.join(bert_ckpt_dir,"bert_config.json")
        MAX_LENGTH=256
        _Review_Classifier_Service._isinstance=_Review_Classifier_Service()
        review_text=_Review_Classifier_Service._isinstance.customizing_input(input_text)
        bert_model = _Review_Classifier_Service._isinstance.create_model(MAX_LENGTH, bert_ckpt_file, bert_config_file,review_text)
        return prediction
"""

input_text="Hey my_name is prakash. I am not just doing a NLP intro."
customized_input=customizing_input(input_text)
print("customized input is",customized_input)
model=create_model(MAX_LENGTH, bert_ckpt_file, bert_config_file)
print(model.summary())
model.compile(
optimizer=keras.optimizers.Adam(1e-5),
loss=keras.losses.BinaryCrossentropy(from_logits=True),
metrics=["accuracy"]
)
model.load_weights("sentiment_bert_weights.h5")
model.save(SAVED_MODEL_PATH)
prediction=model.predict(customized_input)
print("prediction is",prediction)

    