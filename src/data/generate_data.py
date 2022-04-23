
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
  
  
      
def generate_chatbot_data(show_detail: bool):
  words: list[str] = []
  labels: list[str] = []
  doc_x: list[str] = []
  doc_y: list[str] = []
  
  with open(os.path.join(PROJECT_ROOT_PATH, 'data/intents.json'), 'r') as json_data:
    data = json.load(json_data)
    
  for intent in data["intents"]:
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
  
  pickle.dump(words, open(os.path.join(PROJECT_ROOT_PATH, 'models/words.pkl'), 'wb'))
  pickle.dump(labels, open(os.path.join(PROJECT_ROOT_PATH, 'models/labels.pkl'), 'wb'))
  
  X = training[:, 0]
  y = training[:, 1]
  df = pd.DataFrame(data={
    'pattern': X,
    'label': y
  })
  df.to_csv(os.path.join(PROJECT_ROOT_PATH, 'data/chatbot-intents.csv'))
  
  logging.info ('words and labels model created')
  
  return {
    "intents": data,
    'words': words,
    'labels': labels,
    'x_train': np.array(list(training[:,0])),
    'y_train': np.array(list(training[:, 1]))
  }


generate_chatbot_data(show_detail=True)

