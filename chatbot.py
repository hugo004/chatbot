import random
import numpy as np

from typing import Union
from nltk_utils import tokenize, lematize

class Chatbot:
  def __init__(self, words, classes, model, intent_json: object, name='BOT',):
      self.name = name
      self.words = words
      self.classes = classes
      self.model = model
      self.intent_json = intent_json
      
  def bow(self, sentence: Union[list[str], str], detail: bool):
    words = self.words
    sentence_words = self.preprocessing(sentence)
    bag = [0] * len(words)
    for sent_w in sentence_words:
      for i, w in enumerate(words):
        if sent_w == w:
          bag[i] = 1
          if detail:
            print(f'found in bag: ${w}')

    return np.array(bag)

      
  def predict(self, sentence: Union[list[str], str]):
    error_threshold = 0.3
    pattern = self.bow(sentence, detail=False)
    res= self.model.predict(np.array([pattern]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    
    result_list = []
    for r in results:
      result_list.append({ 'intent': self.classes[r[0]], 'probability': r[1] })
    
    return result_list

  def preprocessing(self, sentence: Union[list[str], str]):
    tokenized_words = tokenize(sentence)
    lemmatized_words = [lematize(w.lower()) for w in tokenized_words]
    
    return lemmatized_words

  def get_response(self, intent:object, intent_json: object):
    tag = intent[0]['intent']
    intents = intent_json['intents']
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
      
      intent = self.predict(user_response)
      res = self.get_response(intent, self.intent_json)
      
      print(f'${chatbot_name}: ${res}')