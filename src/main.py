import json
import os
import random

from src.utils import PROJECT_ROOT_PATH
from models.predict import predict_intent, predict_ner

chatbot_name = "DEV"
context = {}
intents_json = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents.json')).read())

def chatbot_response(sentence: str, userId='123', show_detail=False):
  predicted = predict_intent(sentence)
  response = "Sorry, I don't understand"
  
  for result in predicted:
    tag = result['intent']
    intents = intents_json['intents']
    
    for i in intents:
      if i['tag'] == tag:
        if 'context_set' in i:
          if show_detail:
            print('context:', i['context_set'])
            
          context[userId] = i['context_set']
          
        if not 'context_filter' in i or (userId in context and i['context_filter'] == context[userId]):
          if show_detail:
            print('tag:', i['tag'])
            
          response = random.choice(i['responses'])          
          break
  
  return response


if __name__ == '__main__':
  print(f'${chatbot_name}: My name is ${chatbot_name}. What can i help you ?')
  while True:
    user_response = input()
    if user_response in ['bye', 'exit', 'quit']:
      print(f'${chatbot_name}: bye')
      break
    
    response = chatbot_response(user_response)
    ner = predict_ner(user_response)
    
    print('-'*25, 'RESPONSE', '-'*25)
    print(f'${chatbot_name}: ${response}')
    print('-'*25, 'RESPONSE', '-'*25)
    
    print('\n\n')
    print('-'*25, 'NLP', '-'*25)
    doc = ner
    for ent in doc.ents:
      print(ent, ent.text, ent.label_)
    print('-'*25, 'NLP', '-'*25)