import pickle

import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_restful import Api
from tensorflow import keras
from text_classification_with_transformer import PosologyClassificationModel

# Initialize Flask
from text_classification_with_transformer import TokenAndPositionEmbedding

app = Flask(__name__)
api = Api(app)


def load_model(path_to_models='models/keras', vars_file='models/model.pkl'):
    model_file = 'model.h5'
    model = tf.keras.models.load_model(path_to_models)
    modelVars = pickle.load(open(vars_file, 'rb'))
    token = modelVars[0]['token']
    model_posology = modelVars[0]['model']
    return model, token, model_posology


def predict_text(text):
    model, token, model_posology = load_model()
    text = pd.Series([text])
    text_seq = keras.preprocessing.sequence.pad_sequences(token.texts_to_sequences(text), maxlen=model_posology.maxlen)

    prediction = (model.predict(text_seq) > 0.5).astype(int)
    prediction = True if prediction[0, 0] == 1 else False

    return prediction


@app.route("/posology", methods=['GET'])
def posology_api_endpoint():
    global response
    if request.method == "GET":
        requestJson = request.form
        response = {}
        if requestJson and request.form.get("query") is not None:
            query = request.form.get('query')
            response['query'] = query
            prediction = predict_text(query)
            response['is_posology'] = prediction
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=4002)
