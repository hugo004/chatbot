
# import lib
import random
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import gradient_descent_v2
from nltk_utils import lematize


class Model:
  def __init__(self, documents: list[tuple[list[str], str]], words: list[str], classes: list[str]):
      self.documents = documents
      self.words = words
      self.classes = classes
      
  def build(self):
    classes = self.classes
    trainning = []
    output_empty = [0] * len(classes)
    for doc in self.documents:
      bag: list[int] = []
      pattern: list[str] = doc[0]
      pattern = [lematize(w.lower()) for w in pattern]
      
      for w in self.words:
        bag.append(1) if w in pattern else bag.append(0)
      
      output = list(output_empty)
      output[classes.index(doc[1])] = 1
      
      trainning.append([bag, output]) 

    random.shuffle(trainning)
    trainning = np.array(trainning)
    train_x = list(trainning[:,0])
    train_y = list(trainning[:,1])
    
    model = Sequential()
    model.add(Input(shape=(len(train_x[0]),)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    
    SGD = gradient_descent_v2.SGD
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)
    
    print('model created')