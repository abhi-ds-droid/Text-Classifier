# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 08:21:14 2020

@author: Abhishek
"""

import pandas as pd


df=pd.read_csv(r'C:\Users\Abhishek\Desktop\imdb_labelled.csv', header=None, error_bad_lines=False)
df.columns = ['Review','Sentiment']
df=df.dropna()

from sklearn.model_selection import train_test_split

review=df['Review'].values
sentiment=df['Sentiment'].values

review_train, review_test, sentiment_train, sentiment_test= train_test_split(
    review, sentiment, test_size=0.2, random_state=1000)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
vectorizer.fit(review_train)

X_train=vectorizer.transform(review_train)
X_test=vectorizer.transform(review_test)

from sklearn.linear_model import LogisticRegression
classifire=LogisticRegression()
classifire.fit(X_train,sentiment_train)
score=classifire.score(X_test,sentiment_test)
print("Accuracy: ",score)


import tensorflow as tf
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(review_train)

X_train = tokenizer.texts_to_sequences(review_train)
X_test = tokenizer.texts_to_sequences(review_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
from keras.preprocessing.sequence import pad_sequences
maxlen=100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

from keras.models import Sequential
from keras import layers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, sentiment_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(X_test, sentiment_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, sentiment_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, sentiment_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

















