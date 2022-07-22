#!/usr/bin/env python
# coding: utf-8

# # Posology Binary Text classification with Transformer
# 
# **Author:** [Zakaria Mejdoul](https://www.linkedin.com/in/zakaria-mejdoul-225204159/)<br>
# **Date created:** 2022/07/19<br>
# **Last modified:** 2022/07/25<br>
# **Description:** Implement a Transformer block as a Keras layer and use it for posology binary text classification.

# ------------- Setup


import os

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pickle

from metrics_class import Metrics

# progress bar with pandas
tqdm().pandas()

global model_transformers, token, model


# ------------- Implement a Transformer block as a layer


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "ayernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    # ------------- Implement embedding layer

    # Two separate embedding layers, one for tokens, one for token index (positions).


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class PosologyClassificationModel():
    def __init__(self):
        self.TEXT = 'text'
        self.LABEL = 'label'
        self.data_relative_path = "data"
        self.data = None
        self.maxlen = 300

    # ------------- Load and prepare dataset

    @staticmethod
    def load_posology_dataset(self):
        positive_data_path = os.path.join(self.data_relative_path, 'positive_examples_posology.csv')
        negative_data_path = os.path.join(self.data_relative_path, 'negative_examples_posology.csv')

        # Load the training data

        pos_df = pd.read_csv(positive_data_path, delimiter=',', header=None, names=['index', self.TEXT],
                             dtype='str')
        neg_df = pd.read_csv(negative_data_path, delimiter=',', header=None, names=['index', self.TEXT],
                             dtype='str')

        pos_df = pos_df.iloc[1:, :]
        neg_df = neg_df.iloc[1:, :]

        pos_df = pos_df.iloc[:, 1:]
        neg_df = neg_df.iloc[:, 1:]

        pos_df[self.LABEL] = 1
        neg_df[self.LABEL] = 0

        # Concatenate positive and negative dataframes
        full_data = pd.concat([pos_df, neg_df])
        # Shuffle the training data and labels.
        full_data = full_data.sample(frac=1).reset_index(drop=True)

        print(pos_df.head())
        print(neg_df.head())
        print(full_data.head())

        self.data = full_data

        return full_data

    # ------------- Split Data into Training and Test datasets

    def split_data(self):

        self.load_posology_dataset(self)
        x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(self.data[self.TEXT],
                                                                                self.data[self.LABEL],
                                                                                random_state=42,
                                                                                stratify=self.data[self.LABEL],
                                                                                test_size=0.2)

        print(x_train_data.shape)
        print(y_train_data.shape)

        print(x_test_data.shape)
        print(y_test_data.shape)

        return x_train_data, x_test_data, y_train_data, y_test_data

    # ------------- Preprocessing

    # Text Informations

    def text_statistics(self):
        dt = self.load_posology_dataset(self)
        dt['char_count'] = dt[self.TEXT].map(lambda txt: len(str(txt)))  # Number of characters in the string
        dt['word_count'] = dt[self.TEXT].apply(lambda x: len(str(x).split()))  # Number of words in the string
        dt['word_density'] = dt['char_count'] / (dt['word_count'] + 1)  # Density of word (in char)
        print(dt.head())

    def chars_num(self, save=False):
        dt = self.load_posology_dataset(self)
        self.text_statistics()
        # Show the number of characters per text on each row
        plt.figure(figsize=(15, 10))
        max_x = 10000 if dt.char_count.max() > 10000 else dt.char_count.max()
        plt.hist(dt.char_count.values, bins=range(0, max_x, 2))
        plt.title(f"Number of {self.TEXT} {dt.shape[0]}")
        plt.xlabel("Number of characters")
        plt.ylabel("Number of documents")
        plt.grid(True)
        plt.show()
        if save:
            plt.savefig("numb_char.png")

    # ------------- histogram of the class frequency
    def hist_class_frequency(self, save=False):
        dt = self.load_posology_dataset(self)
        dt[self.LABEL].hist(xrot=45)
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        plt.title("Class Frequency")
        plt.show()
        if save:
            plt.savefig("distrib_classes.png")

    # ------------- Label Encoding

    @staticmethod
    def encode_label(y_train, y_test):
        # label encode the target variable
        encoder = sklearn.preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(y_train)
        test_y = encoder.fit_transform(y_test)
        return train_y, test_y

    # ------------- Word Embeddings

    def tokenize_words(self, x_train, x_test, y_train, y_test):

        # create a tokenizer
        token = Tokenizer()
        token.fit_on_texts(self.data[self.TEXT].astype('str'))
        word_index = token.word_index

        print(len(x_train), "Training sequences")
        print(len(x_test), "Validation sequences")
        x_train_seq = keras.preprocessing.sequence.pad_sequences(token.texts_to_sequences(x_train.astype('str')),
                                                                 maxlen=self.maxlen)
        x_test_seq = keras.preprocessing.sequence.pad_sequences(token.texts_to_sequences(x_test.astype('str')),
                                                                maxlen=self.maxlen)

        print(x_train_seq)
        print(y_train)
        print(x_test_seq)
        print(y_test)

        return word_index, x_train_seq, x_test_seq, token

    # ------------- Create classifier model using transformer layer


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "projection_dim": self.projection_dim,
            "query_dense": self.query_dense,
            "key_dense": self.key_dense,
            "value_dense": self.value_dense,
            "combine_heads": self.combine_heads
        })
        return config

    @staticmethod
    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


def transformers_classifier(word_index, label=None):
    '''
    Function to generate a rcnn for binary or multiclass classification.
    '''
    if label is None:
        label = [0, 1]
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = len(word_index) + 1
    maxlen = 300
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(1 if len(label) <= 2 else len(label),
                                 activation='sigmoid' if len(label) <= 2 else "softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    if len(label) == 2:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    print(model.summary())

    return model


def build_model():
    model = PosologyClassificationModel()
    x_train_data, x_test_data, y_train_data, y_test_data = model.split_data()
    word_index, x_train_seq, x_test_seq, token = model.tokenize_words(x_train_data, x_test_data, y_train_data,
                                                                      y_test_data)
    train_y, test_y = model.encode_label(y_train_data, y_test_data)
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    vocab_size = len(word_index) + 1
    inputs = layers.Input(shape=(model.maxlen,))
    embedding_layer = TokenAndPositionEmbedding(model.maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model_transformers = transformers_classifier(word_index, label=[0, 1])

    return model_transformers, x_train_seq, x_test_seq, train_y, test_y, token, model

    # ------------- Train and Evaluate


def train():
    model_transformers, x_train_seq, x_test_seq, train_y, test_y, token, model = build_model()
    # Early Stopping & Model saving
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=3)
    callbacks = tf.keras.callbacks.ModelCheckpoint("models/model.h5", save_best_only=True)

    history = model_transformers.fit(
        x_train_seq, train_y,
        epochs=50,
        validation_split=0.2, batch_size=32)
    return history, model_transformers, x_test_seq, test_y, token, model


def evaluate(model_transformers, x_test_seq, test_y):
    results = model_transformers.evaluate(x_test_seq, test_y)
    print(results)
    print(
        f"\nThe precision of the model is {round(100 * precision_score(test_y, (model_transformers.predict(x_test_seq) > 0.5).astype(int), labels=[0, 1]), 2)}%")

    # ------------- Displaying Metrics


def display_metrics(model_transformers, history, x_test_seq, test_y):
    metric = Metrics()
    metric.metrics_deep_learning(model_transformers, history, x_test_seq, test_y, ["negative", "positive"])


def predict_text(model_transformers, text, token, model):
    text = pd.Series([text])
    text_seq = keras.preprocessing.sequence.pad_sequences(token.texts_to_sequences(text), maxlen=model.maxlen)

    prediction = (model_transformers.predict(text_seq) > 0.5).astype(int)
    prediction = True if prediction[0, 0] == 1 else False

    return prediction


if __name__ == "__main__":
    history, model_transformers, x_test_seq, test_y, token, model = train()
    evaluate(model_transformers, x_test_seq, test_y)
    display_metrics(model_transformers, history, x_test_seq, test_y)
    prediction = predict_text(model_transformers, "3 mois", token, model)
    PATH_TO_MODELS = 'models/'
    filename = 'model.pkl'
    model_vars = {
        'token': token,
        'model': model
    }
    model_transformers.save('models/keras')
    pickle.dump([model_vars], open(PATH_TO_MODELS + filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
