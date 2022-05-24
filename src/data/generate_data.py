
#!/usr/bin/env python3
import logging
import os
import random
import string
import json

import pickle
import nltk
import numpy as np
import pandas as pd


# from utils.nltk_utils import tokenize, lematize
from typing import Union
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from src.utils import PROJECT_ROOT_PATH

# # run at first time
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')
logging.basicConfig(level=logging.INFO)



lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lematize(word: str):
    return lemmatizer.lemmatize(word=word.lower())
  
def tokenize(word: str):
    return nltk.word_tokenize(word)

def stem(word: str):
    return stemmer.stem(word=word.lower())



def extract_word_info(sentence: str, word: Union[str,list[str]], label: str):
  def get_info(sentence, word) -> tuple:
    try:
      start = sentence.index(word)
      end = len(word) + start
      found = sentence[start:end]
      logging.info(f'start: {start} end: {end} -> {found}')
      
      return (start,end, label)
    except:
      return None
    
  if isinstance(word, list):
    return [get_info(sentence, w) for w in word]
  else:
    return get_info(sentence, word)
  
  
def generate_ner_data():
  training_data = []
  data = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/ner-train-raw.json')).read())
  
  for text in data['texts']:
    entities = []
    for ner in data['ners']:
      info = extract_word_info(text, ner['text'], ner['label'])
      if info is not None:
        entities.append({
          'start': info[0],
          'end': info[1],
          'label': info[2]
        })
    
    if len(entities) > 0:
      training_data.append({
        'text': text,
        'entities': entities
      })
    
  with open(os.path.join(PROJECT_ROOT_PATH, 'data/ner-train-data.json'), 'w') as f:
    json.dump(obj={ 'texts': training_data }, fp=f, indent=2,)
    
  logging.info('NER data generated')
  
  
  

def get_intents():
  general_intent = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/general-intents.json')).read())
  booking_intent = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/booking-intents.json')).read())
  eform_intent = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/eform-intents.json')).read())
  
  intents = []
  intents.extend(general_intent['intents'])
  intents.extend(booking_intent['intents'])
  intents.extend(eform_intent['intents'])
  
  return intents
  
      
def generate_chatbot_data(show_detail: bool, name="chatbot-intents", intents:list = get_intents()):
  words: list[str] = []
  labels: list[str] = []
  doc_x: list[str] = []
  doc_y: list[str] = []
  
  # intents =  get_intents()
 
    
  for intent in intents:
    for pattern in intent["patterns"]:
      tokenized = tokenize(pattern)
      words.extend(tokenized)
      doc_x.append(pattern)
      doc_y.append(intent["tag"])
      
      if intent["tag"] not in labels:
        labels.append(intent["tag"])
        if show_detail:
          logging.info(f'append label -> {intent["tag"]}')
        
  words = [lematize(w) for w in words if w not in string.punctuation]
  words = sorted(set(words))
  labels = sorted(set(labels))
  
  training = []
  output_empty = [0] * len(labels)
  for idx, doc in enumerate(doc_x):
    bow = []
    text = lematize(doc)
    if show_detail:
      logging.info(f'text -> {text}')
    
    for word in words:
      bow.append(1) if word in text else bow.append(0)
      if show_detail:
        logging.info(f'append word -> {word}')
      
    output_row = list(output_empty)
    output_row[labels.index(doc_y[idx])] = 1
    training.append([bow, output_row])
    
  random.shuffle(training)
  training = np.array(training, dtype=object)
  
  pickle.dump(words, open(os.path.join(PROJECT_ROOT_PATH, f'models/{name}-words.pkl'), 'wb'))
  pickle.dump(labels, open(os.path.join(PROJECT_ROOT_PATH, f'models/{name}-labels.pkl'), 'wb'))
  

  df = pd.DataFrame(data={
    "pattern": training[:,0],
    "label": training[:,1]
  })
  df.to_csv(os.path.join(PROJECT_ROOT_PATH, f'data/{name}.csv'))

  logging.info ('words and labels model created')
  
  return {
    "intents": intents,
    'words': words,
    'labels': labels,
    'x_train': np.array(list(training[:,0])),
    'y_train': np.array(list(training[:, 1])),
  }


