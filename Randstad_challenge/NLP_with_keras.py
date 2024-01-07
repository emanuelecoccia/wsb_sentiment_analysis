import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

train_df = pd.read_csv('Dataset_Randstad-Challenge/train_set.csv')
test_df = pd.read_csv('Dataset_Randstad-Challenge/test_set.csv')

x_train, y_train = train_df['Job_offer'], train_df['Label']
x_test, y_test = test_df['Job_offer'], test_df['Label']

labels = {'Java Developer':0, 'Software Engineer':1, 'Programmer':2, 'System Analyst':3, 'Web Developer':4}
y_train = np.array([labels[label] for label in y_train], dtype=np.float32)
y_test = np.array([labels[label] for label in y_test], dtype=np.float32)

# creating a tokenizer object
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 3000, oov_token=True)

# tokenizing the training data 
# (not the test data, otherwise it would make no sense)
tokenizer.fit_on_texts(x_train)

# now tranforming words into integers
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# and integers into binary values (multi-hot encoding)
x_train = tokenizer.sequences_to_matrix(train_sequences, mode='binary')
x_test = tokenizer.sequences_to_matrix(test_sequences, mode='binary')

# keras.utils.to_categorical converts a class vector (integers) to binary class matrix.
# Since, in this case, we only have one integer per label,
# it becomes a one-hot encoding.
y_train = tf.keras.utils.to_categorical(y_train,num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test,num_classes=5)

# creating the neural net
model = Sequential()
model.add(tf.keras.layers.Dense(128,input_shape=(x_train[0].shape)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# compiling the model and fitting on our data
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy', metrics.Precision, metrics.Recall, ])
model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=64,epochs=100)

# how did we score?
model.evaluate(x_test,y_test)
