import json
import os
import numpy as np
import spacy
import pickle

from keras.models import load_model
from typing import Union
from src.utils import PROJECT_ROOT_PATH
from src.utils.nltk_utils import tokenize, bow


chatbot_model = load_model(os.path.join(PROJECT_ROOT_PATH, 'models/chatbot_model.h5'))
ner_nlp = spacy.load(os.path.join(PROJECT_ROOT_PATH, 'models/ner/model-extend'))
words = pickle.load(open(os.path.join(PROJECT_ROOT_PATH, 'models/words.pkl'), 'rb'))
labels = pickle.load(open(os.path.join(PROJECT_ROOT_PATH, 'models/labels.pkl'), 'rb'))

threshold = 0.5

def preprocess_sentence(sentence: Union[list[str], str]):
  # tokenized
  tokenized_sentence = tokenize(sentence)
  bag = bow(tokenized_sentence, words)
  
  return bag

def predict_intent(sentence: str):
  sentence = preprocess_sentence(sentence)
  results = chatbot_model.predict(np.array([sentence]))[0]
  results = [[i, r] for i, r in enumerate(results) if r > threshold]
  results.sort(key=lambda x : x[1], reverse=True)
  
  intent_list = []
  for result in results:
    intent_list.append({
      'intent': labels[result[0]],
      'probability': result[1]
    })
    
  return intent_list

def predict_ner(sentence: str):
  doc = ner_nlp(sentence)
  return doc