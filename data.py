from keras.datasets import imdb
from keras.preprocessing import sequence

top_words = 5000
max_review_length = 500
start_index = 1
oov_index = 2


def get_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words, oov_char=oov_index)
    x_train = sequence.pad_sequences(x_train, maxlen=max_review_length, truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_review_length, truncating='post')

    return x_train, y_train, x_test, y_test
