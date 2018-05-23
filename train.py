import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.utils import plot_model

import data

x_train, y_train, x_test, y_test = data.get_data()

print('x_train shape: ' + str(x_train.shape))
print('x_test shape: ' + str(x_test.shape))
print('y_train shape: ' + str(y_train.shape))
print('y_test shape: ' + str(y_test.shape))

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(input_length=data.max_review_length, input_dim=data.top_words, output_dim=embedding_vector_length))
model.add(LSTM(units=100))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

epoch = keras.callbacks.ModelCheckpoint('epoch{epoch:02d}_model.hdf5')
best = keras.callbacks.ModelCheckpoint('best_model.hdf5', save_best_only=True, monitor='val_loss')
model.fit(x_train, y_train, nb_epoch=3, batch_size=128, validation_split=0.1, callbacks=[epoch, best])

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: ' + str(loss) + ', test accuracy: ' + str(accuracy))
