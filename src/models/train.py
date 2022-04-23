import os
import logging
import spacy
import json

from utils import PROJECT_ROOT_PATH
from spacy.tokens import DocBin
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import gradient_descent_v2


def train_ner_model(show_detail = False):
  '''
  train custom NER
  '''
  train_data = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/ner-train-data.json')).read())
  nlp = spacy.blank('en')
  
  db = DocBin()
  for data in train_data['texts']:
    text = data['text'].lower()
    entities = data['entities']
    
    doc = nlp.make_doc(text)
    if show_detail:
        logging.info('training -> text:', text, 'entities:', entities)
      
    ents = []
    for ent in entities:
      start, end, label  = ent['start'], ent['end'], ent['label']
      
      span = doc.char_span(start, end, label=label.upper())
      if span is None:
        logging.warn('skipping entity')
      else:
        ents.append(span)
        if show_detail:
          logging.info('append -> ', span)
    doc.ents = ents
    db.add(doc)
    
  db.to_disk('./models/ner/train.spacy')
  logging.info('NER model saved')


def train_chatbot_model(X_train, y_train):
  '''
  train chatbot model
  '''
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
  model.save(os.path.join(PROJECT_ROOT_PATH, 'models/chatbot_model.h5'), hist)
  
  logging.info('chatbot model saved')