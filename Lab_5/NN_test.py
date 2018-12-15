from keras import Sequential
from keras.datasets import imdb
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence

vocabulary_size = 5000

(x, y), (_x, _y) = imdb.load_data(num_words=vocabulary_size)
print('loaded dataset with {} training samples, {} test samples'.format(len(x), len(_x)))

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in x[6]])
print('---label---')
print(y[6])

print('Maximum review length: {}'.format(
    len(max((x + _x), key=len))))

print('Minimum review length: {}'.format(
    len(min((x + _x), key=len))))

max_words = 500
x = sequence.pad_sequences(x, maxlen=max_words)
_x = sequence.pad_sequences(_x, maxlen=max_words)

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = x[:batch_size], y[:batch_size]
X_train2, y_train2 = x[batch_size:], y[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(_x, _y, verbose=0)
print('Test accuracy:', scores[1])
