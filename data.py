
import random
import string
import json
import numpy as np

from nltk_utils import tokenize, lematize


      
def get_data():
  words: list[str] = []
  labels: list[str] = []
  doc_x: list[str] = []
  doc_y: list[str] = []
  
  with open('intents.json', 'r') as json_data:
    data = json.load(json_data)
    
  for intent in data["intents"]:
    for pattern in intent["patterns"]:
      tokenized = tokenize(pattern)
      words.extend(tokenized)
      doc_x.append(pattern)
      doc_y.append(intent["tag"])
      
      if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
  words = [lematize(w) for w in words if w not in string.punctuation]
  words = sorted(set(words))
  labels = sorted(set(labels))
  
  training = []
  output_empty = [0] * len(labels)
  for idx, doc in enumerate(doc_x):
    bow = []
    text = lematize(doc)
    for word in words:
      bow.append(1) if word in text else bow.append(0)
      
    output_row = list(output_empty)
    output_row[labels.index(doc_y[idx])] = 1
    training.append([bow, output_row])
    
  random.shuffle(training)
  training = np.array(training)
  
  return {
    "intents": data,
    'words': words,
    'labels': labels,
    'x_train': np.array(list(training[:,0])),
    'y_train': np.array(list(training[:, 1]))
  }
