import random
import numpy as np

from typing import Union
from nltk_utils import tokenize, lematize, stem
from tensorflow.keras.models import load_model
from data import get_data

class Chatbot:
  def __init__(self, name='BOT',):
      self.name = name
      self.model = load_model('chatbot_model.h5')
      
      data = get_data()
      self.words = data['words']
      self.labels = data['labels']
      self.intents = data['intents']
      
  def bow(self, tokenized_sentence: Union[list[str], str], words: list[str]):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
      if w in sentence_words:
        bag[idx] = 1
      
    return bag

      
  def predict(self, sentence: Union[list[str], str]):
    error_threshold = 0.7
    pattern = sentence
    res= self.model.predict(np.array([pattern]))
    predicted = np.argmax(res)
    prob = max(res[0])
    
    if prob > error_threshold:
      return { 'intent': self.labels[predicted], 'probability': prob }

    return None


  def preprocessing(self, sentence: Union[list[str], str]):
    # tokenized
    tokenized_sentence = tokenize(sentence)
    bag = self.bow(tokenized_sentence, self.words)
    
    return bag

  def get_response(self, intent:object, intents: object):
    if intent == None:
      return 'Sorry, i do not understand'
      
    tag = intent['intent']
    intents = intents['intents']
    for i in intents:
      if i['tag'] == tag:
        result = random.choice(i['responses'])
        break
    
    return result

      
  def start_chat(self):
    chatbot_name = self.name
    print(f'${chatbot_name}: My name is ${chatbot_name}. What can i help you ?')
    
    while(True):
      user_response = input()
      user_response = user_response.lower()
      if user_response in ['bye', 'exit', 'quit']:
        print(f'${chatbot_name}: bye')
        break
      
      proccesed = self.preprocessing(user_response)
      intent = self.predict(proccesed)
      res = self.get_response(intent, self.intents)
      
      print(f'${chatbot_name}: ${res}')