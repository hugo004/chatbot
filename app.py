
# import lib
import nltk
# run at first time
# nltk.download('punkt')
# nltk.download('wordnet')

import json
import numpy as np

from tensorflow.keras.models import load_model
from model import Model
from chatbot import Chatbot
from nltk_utils import tokenize

# init training of chatbot
words: list[str] = []
classes: list[str] = []
documents: list[(list[str],str)] = []
ignore_words = ['?', '!']

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

for intent in intents["intents"]:
  for pattern in intent["patterns"]:
    w = tokenize(pattern)
    words.extend(w)
    documents.append((w, intent["tag"]))
    
    if intent["tag"] not in classes:
      classes.append(intent["tag"])


if __name__ == '__main__':
  # # build model
  model = Model(documents=documents, words=words, classes=classes)
  model.build()
  
  # stat chat
  model = load_model('chatbot_model.h5')
  chatbot = Chatbot(words=words, classes=classes, model=model, intent_json=intents)
  chatbot.start_chat()