import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.text import text_to_word_sequence

max_features = 20000
max_len = 30
batch_size = 32

(xtrain, ytrain), (xtest, ytest) = imdb.load_data(num_words=max_features)
print(len(xtrain), 'Train Amount')
print(len(xtest), 'Test Amount')

# Set the amount of words per sequence
xtrain = sequence.pad_sequences(xtrain, maxlen=max_len)
xtest = sequence.pad_sequences(xtest, maxlen=max_len)

# One data entry has 30 "words" per sequence.
print(xtest[1])

# Shape of the dataset for xtrain and xtest
print(xtrain.shape, 'Train Shape')
print(xtest.shape, 'Test Shape')

# Initialize the RNN

rnn_hidden_dim = 5
word_embedding_dim = 50

modelrnn = Sequential()
modelrnn.add(keras.Input(shape=(max_len,)))
modelrnn.add(Embedding(max_features, word_embedding_dim))
modelrnn.add(SimpleRNN(
    rnn_hidden_dim,
    kernel_initializer=initializers.RandomNormal(stddev=0.001),
    recurrent_initializer=initializers.Identity(gain=0.1),
    activation="tanh",
))

modelrnn.add(Dense(1,activation='sigmoid'))
modelrnn.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
modelrnn.compile(loss='binary_crossentropy',
                 optimizer=rmsprop,
                 metrics=['accuracy'])

modelrnn.fit(xtrain, ytrain,
             batch_size=batch_size,
             epochs=10,
             validation_data=(xtest,ytest))

score, acc = modelrnn.evaluate(xtest,ytest,
                               batch_size=batch_size)

print("First Epoch Batch Test Score", score)
print("First Epoch Batch Test Accuracy", acc)

modelrnn.fit(xtrain, ytrain,
             batch_size=batch_size,
             epochs=10,
             validation_data=(xtest,ytest))

score, acc = modelrnn.evaluate(xtest,ytest,
                               batch_size=batch_size)

print("Second Epoch Batch Test Score", score)
print("Second Epoch Batch Test Accuracy", acc)

word_index = imdb.get_word_index()
word_index = {word: index + 3 for word, index in word_index.items()}

def predict_review(review):
    words = text_to_word_sequence(review)
    tokens = [1] + [word_index.get(word, 2) for word in words]
    tokens = [token if token < max_features else 2 for token in tokens]

    padded = sequence.pad_sequences([tokens], maxlen=max_len)
    positive_probability = float(modelrnn.predict(padded, verbose=0)[0, 0])

    sentiment = "positive" if positive_probability >= 0.5 else "negative"
    print(f"{sentiment} (positive probability: {positive_probability:.3f})")

while True:
    review = input("Enter a review (or 'quit' to stop): ")
    if review.lower() == "quit":
        break
    predict_review(review)



