import random
import numpy as np

from typing import Union
from nltk_utils import tokenize, lematize, stem
from tensorflow.keras.models import load_model
from data import get_data

context = {}
detail = True
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
    error_threshold = 0.5
    pattern = sentence
    res= self.model.predict(np.array([pattern]))[0]
    res = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    res.sort(key=lambda x: x[1], reverse=True)
    
    result_list = []
    for i in res:
      result_list.append({ 'intent': self.labels[i[0]], 'probability': i[1] })
      
    return result_list
    
    # predicted = np.argmax(res)
    # prob = max(res[0])
    
    # if detail:
    #   print('predict:', { 'intent': self.labels[predicted], 'probability': prob })
    
    # if prob > error_threshold:
    #   return { 'intent': self.labels[predicted], 'probability': prob }

    # return None


  def preprocessing(self, sentence: Union[list[str], str]):
    # tokenized
    tokenized_sentence = tokenize(sentence)
    bag = self.bow(tokenized_sentence, self.words)
    
    return bag

  def get_response(self, sentence, intents: object, userId='123'):
    predicted = self.predict(sentence)
    resposne = 'Sorry, i do not understand'
    
    for result in predicted:
      tag = result['intent']
      intents = intents['intents']
      
      for i in intents:
        if i['tag'] == tag:
          if 'context_set' in i:
            if detail:
              print('context:', i['context_set'])
              
            context[userId] = i['context_set']
            
          if not 'context_filter' in i or (userId in context and i['context_filter'] == context[userId]):
            if detail:
              print('tag:', i['tag'])
              
            resposne = random.choice(i['responses'])          
            break
    
    return resposne

      
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
      res = self.get_response(proccesed, self.intents)
      
      print(f'${chatbot_name}: ${res}')