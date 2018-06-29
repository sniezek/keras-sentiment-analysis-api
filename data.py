from keras.datasets import imdb
from keras.preprocessing import sequence

top_words = 5000
max_review_length = 500
start_index = 1
oov_index = 2
offset = 3


def get_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words, oov_char=oov_index)
    x_train = pad(x_train)
    x_test = pad(x_test)

    return x_train, y_train, x_test, y_test


def pad(data):
    return sequence.pad_sequences(data, maxlen=max_review_length, truncating='pre')