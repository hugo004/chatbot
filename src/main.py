from cProfile import label
import json
import os
import random

from src.utils import PROJECT_ROOT_PATH
from models.predict import predict_intent, predict_ner

intents_json = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents.json')).read())
chatbot_name = "DEV"
context = {}
scene_context = {}
user = '123'

def chatbot_response(sentence: str, userId, show_detail=False):
  ner = predict_ner(sentence)
  predicted = predict_intent(sentence)
  response = "Sorry, I don't understand"
  user_intent = None
  
  if show_detail:
    print('predicted:', predicted)
  
  for result in predicted:
    tag = result['intent']
    intents = intents_json['intents']
    
    previous_context = context[userId] if userId in context else '' 
    for i in intents:
      if i['tag'] == tag:
        if 'context_set' in i:
          if show_detail:
            print('context:', i['context_set'], 'prvious:', previous_context)
          
          context[userId] = i['context_set']
          

        if not 'context_filter' in i or ('context_filter' in i and 'context_set' in i) or (userId in context and i['context_filter'] == context[userId]):
          if show_detail:
            print('tag:', i['tag'])
            
          response = random.choice(i['responses'])     
          user_intent = i['tag'] 
          
        if 'context_filter' in i:
          print('\n\n')
          print('-'*25, 'NLP', '-'*25)
          print('predicted user intent: ', user_intent)
          
          doc = ner
          for ent in doc.ents:
            if i['context_filter'] == 'bkd-query' and  ent.label_ == 'destination'.upper():
                scene_context['destination'] = ent.text
            elif i['context_filter'] == 'bkt-query':
              if ent.label_ == 'depature'.upper():
                scene_context['depature'] = ent.text
              elif ent.label == 'return'.upper():
                scene_context['return'] = ent.text
              
            print(ent, ent.text, ent.label_)
          
          break
  
  return (response, user_intent)


if __name__ == '__main__':
  print(f'${chatbot_name}: My name is ${chatbot_name}. What can i help you ?')
  while True:
    user_response = input()
    if user_response in ['bye', 'exit', 'quit']:
      print(f'${chatbot_name}: bye')
      print('\n')
      print('-'*10, 'booking summary', '-'*10)
      print(scene_context)
      break
    
    response, user_intent = chatbot_response(user_response, userId=user, show_detail=True)
    
    print('-'*25, 'RESPONSE', '-'*25)
    print(f'${chatbot_name}: ${response}\n')
    