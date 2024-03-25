import re
import pickle
import numpy as np
import pandas as pd

from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

class CustomFlaskAppWithEncoder(Flask):
    json_provider_class = LazyJSONEncoder

app = CustomFlaskAppWithEncoder(__name__)

swagger_template = dict(
    info = {
        'title' : LazyString(lambda: "API For Sentiment Prediction, By Group 3"),
        'version' : LazyString(lambda: "1.0.0"),
        'description' : LazyString(lambda: "API untuk Prediksi Sentimen, Oleh Kelompok 3"),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers" : [],
    "specs" : [
        {
            "endpoint": "docs",
            "route" : "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                  config = swagger_config)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

def cleansing(sent):
    #mengubah menjadi huruf kecil semua
    string = sent.lower()
    # menghapus alamat URL
    url = re.compile(r'https?://\S+|www\.\S+')
    string = url.sub(r'', string)

    # hapus emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    string = emoji_pattern.sub(r'', string)

    # hapus nomor
    string = re.sub(r'\d+', '', string)

    # hapus simbol
    string = re.sub(r'[^a-zA-Z0-9\s]', '', string)

    return string

#load Feature Extraction LSTM
file_lstm = open("Challange-platinum-kelompok3/resources_of_lstm/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file_lstm)
file_lstm.close()

#Load Model LSTM
model_file_from_lstm = load_model('Challange-platinum-kelompok3/model_of_lstm/model.h5')

def predict_sentiment_LSTM(text):
    cleaned_text = cleansing(text)
    input_text = [cleaned_text]

    predicted = tokenizer.texts_to_sequences(input_text)
    guess = pad_sequences(predicted, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(guess)[0]
    sentiment_label = sentiment[np.argmax(prediction)]

    return sentiment_label

def predict_sentiment_neural_network(text):
    cleaned_text = cleansing(text)
    input_text = [cleaned_text]

    input_vector = count_vect.transform(input_text)
    sentiment_label = loaded_model.predict(input_vector)[0]

    return sentiment_label

#Load Model NeuralNetwork
count_vect = CountVectorizer()
count_vect = pickle.load(open("Challange-platinum-kelompok3/resources_of_nn/feature_platinum_nn.p", 'rb'))
loaded_model = pickle.load(open("Challange-platinum-kelompok3/model_of_nn/model_nn.p", 'rb'))

#endpoint LSTM Teks
@swag_from("docs/lstm.yml", methods = ['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Hasil Analisis Sentimen Menggunakan LSTM",
        'data' : {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

#Endpoint LSTM File
#Proses Load File Data Tweetnya LAMA BANGET(Mohon ditunggu - bisa nyampe 15-20 menitan)
@swag_from("docs/LSTMfile.yml", methods=['POST'])
@app.route('/Upload File LSTM', methods=['POST'])
def LSTM_upload_file():
    file = request.files["upload_file"]
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df['Tweet_Clean'] = df['Tweet'].apply(cleansing)
    df['Sentiment'] = df['Tweet_Clean'].apply(predict_sentiment_LSTM)

    sentiment_results = df[['Tweet_Clean', 'Sentiment']].to_dict(orient='records')
    return jsonify({'output': sentiment_results})

# #endpoint Neural Network
@swag_from("docs/nn.yml", methods = ['POST'])
@app.route('/Neural Network', methods=['POST'])
def nn():
    original_text = request.form.get('text')
    text = count_vect.transform([cleansing(original_text)])
    
    sentiment = loaded_model.predict(text)[0]

    json_response = {
        'status_code' : 200,
        'description' : "Hasil Analisis Sentimen Menggunakan Neural Network",
        'data' : {
            'text': original_text,
            'sentiment': sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# endpoint Upload File Neural Network
@swag_from("docs/NNfile.yml", methods= ['POST'])
@app.route('/Upload File Neural Network', methods= ['POST'])
def NN_upload_file():
    file = request.files["upload_file"]
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df['Tweet_Clean'] = df['Tweet'].apply(cleansing)
    df['Sentiment'] = df['Tweet_Clean'].apply(predict_sentiment_neural_network)

    sentiment_results = df[['Tweet_Clean', 'Sentiment']].to_dict(orient='records')
    return jsonify({'output': sentiment_results})


if __name__ == '__main__':
    app.run()
   
