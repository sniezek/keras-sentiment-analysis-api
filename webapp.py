import keras
from flask import Flask, request, render_template
from keras.backend import tf
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence

import data

app = Flask(__name__)
model = None
words_to_indices = imdb.get_word_index()


def load_model():
    global model
    model = keras.models.load_model('saved_model.hdf5')
    global graph
    graph = tf.get_default_graph()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_sentiment():
    text = request.get_json(force=True)['text']
    words = text_to_word_sequence(text)
    word_indices = [words_to_indices[word] + data.offset if word in words_to_indices and words_to_indices[word] < data.top_words else data.oov_index for word in words]
    sequence = data.pad([[data.start_index] + word_indices])
    with graph.as_default():
        sentiment = model.predict(sequence)
    return '{0:.2f}% positive'.format(sentiment[0][0] * 100)


if __name__ == '__main__':
    load_model()
    app.run()
