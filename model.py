
# import lib
from statistics import mode
import nltk
# run at first time
# nltk.download('punkt')
# nltk.download('wordnet')

import json
import pickle
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import gradient_descent_v2

# init training of chatbot
words: list[str] = []
classes: list[str] = []
documents: list[(list[str],str)] = []
ignore_words = ['?', '!']

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

for intent in intents["intents"]:
  for pattern in intent["patterns"]:
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    documents.append((w, intent["tag"]))
    
    if intent["tag"] not in classes:
      classes.append(intent["tag"])
  
  
# build chatbot model
trainning = []
output_empty = [0] * len(classes)
for doc in documents:
  bag: list[int] = []
  pattern: list[str] = doc[0]
  pattern = [lemmatizer.lemmatize(w.lower()) for w in pattern]
  
  for w in words:
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
model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

SGD = gradient_descent_v2.SGD
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('model created')

# connect telegram