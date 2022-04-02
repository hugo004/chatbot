import json
import random
from unittest import result
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from typing import Union

def preprocessing(sentence: Union[list[str], str]):
  tokenize = nltk.word_tokenize(sentence)
  lemmatize = [lemmatizer.lemmatize(w.lower()) for w in tokenize]
  
  return lemmatize


def bow(sentence: Union[list[str], str], words: list[str], detail: bool):
  sentence_words = preprocessing(sentence)
  bag = [0] * len(words)
  for sent_w in sentence_words:
    for i, w in enumerate(words):
      if sent_w == w:
        bag[i] = 1
        if detail:
          print(f'found in bag: ${w}')

  return np.array(bag)


def predict(sentence: Union[list[str], str], words: list[str], classess: list[str], model:Sequential):
  error_threshold = 0.3
  pattern = bow(sentence, words, detail=False)
  print(pattern, '\n\n\n')
  res= model.predict(np.array([pattern]))[0]
  results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
  
  result_list = []
  for r in results:
    result_list.append({ 'intent': classess[r[0]], 'probability': r[1] })
  
  return result_list


def get_response(intent:object, intent_json: object):
  tag = intent[0]['intent']
  intents = intent_json['intents']
  for i in intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  
  return result


chatbot_name = 'Berserker'
model = load_model('chatbot_model.h5')

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
  
  
    
def start_chat():
  print(f'${chatbot_name}: My name is ${chatbot_name}. What can i help you ?')
  while(True):
    user_response = input()
    user_response = user_response.lower()
    if user_response in ['bye', 'exit', 'quit']:
      print(f'${chatbot_name}: bye')
      break
    
    intent = predict(user_response, words=words, classess=classes, model=model)
    res = get_response(intent, intents)
    
    print(f'${chatbot_name}: ${res}')


start_chat()