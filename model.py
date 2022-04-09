
# import lib
import random
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import gradient_descent_v2
from nltk_utils import lematize, stem


class Model:
  def __init__(self):
    pass
      
  def fit(self, X_train, y_train):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(y_train[0]), activation='softmax'))

    
    SGD = gradient_descent_v2.SGD
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)
    
    return model
    