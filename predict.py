import random

import keras
import numpy as np
from keras.datasets import imdb

import data

words_to_indices = imdb.get_word_index()
indices_to_words = {v: k for k, v in words_to_indices.items()}

model = keras.models.load_model('saved_model.hdf5')
model.summary()

_, _, x_test, y_test = data.get_data()

while True:
    random_review_index = random.randint(0, len(x_test) - 1)

    word_indices = x_test[random_review_index]
    sentiment = y_test[random_review_index]

    actual_word_indices = [word_index - 3 for word_index in word_indices]  # because of start_index and oov_index
    words = [indices_to_words[index] for index in actual_word_indices if index in indices_to_words]
    print(*words)
    print('Sentiment: ' + 'Positive(1)' if sentiment == 1 else 'Negative(0)')

    print(model.predict(np.array([word_indices])))
    print()
