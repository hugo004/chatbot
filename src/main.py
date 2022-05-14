import json
import os
import random

from src.utils import PROJECT_ROOT_PATH
from models.predict import predict_intent, predict_ner

intents_json = json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents.json')).read())
chatbot_name = "DEV"
context = {}
scene_context = {}
bot_memory = []
user = '123'

def catch_ner_info(ent, context_filter, show_detail = False):
  ner_tag = ent.label_
  if ner_tag in ['FROM', 'TO', 'DPTL', "DSTT", 'RTND', 'DPTD']:
    bot_memory.append(ner_tag)
  
  latest_memory = bot_memory[-1] if len(bot_memory) > 0 else None
  
  if show_detail:
    print('current ner', ner_tag,'bot memory:', bot_memory, 'latest memory:', latest_memory)
    
  if context_filter == 'bkd-query':
    if ner_tag == 'GPE':
      if latest_memory in ['FROM', 'DPTL']:
        scene_context['destination'] = ent.text
        bot_memory.pop()
      elif latest_memory in ['TO', 'DSTT']:
        scene_context['depature'] = ent.text
        bot_memory.pop()
    
      
    
      
  elif context_filter in ['bkt-query', 'bklt-query']:
    if ner_tag in ['CARDINAL', 'DATE']:
      if latest_memory in ['FROM', 'DPTD']:
        scene_context['return-date'] = ent.text
        bot_memory.pop()
      elif latest_memory in ['TO', 'RTND']:
        scene_context['depature-date'] = ent.text
        bot_memory.pop()


def chatbot_response(sentence: str, userId, show_detail=False):
  ner = predict_ner(sentence)
  predicted = predict_intent(sentence)
  response = "Sorry, I don't understand"
  user_intent = None
  
  if show_detail:
    print('\npredicted:', predicted)
  
  for result in predicted:
    tag = result['intent']
    intents = intents_json['intents']
    
    for i in intents:
      if i['tag'] == tag:
        if 'context_set' in i:
          if show_detail:
            print('context:', i['context_set'])
          
          context[userId] = i['context_set']
          

        if not 'context_filter' in i or ('context_filter' in i and 'context_set' in i) or (userId in context and i['context_filter'] == context[userId]):
          if show_detail:
            print('tag:', i['tag'])
            
          response = random.choice(i['responses'])     
          user_intent = i['tag'] 
          
        if 'context_filter' in i:
          print('\n')
          print('-'*25, 'NLP', '-'*25)
          print('predicted user intent: ', user_intent)
          
          doc = ner
          print('nlp ents:', doc.ents)
          for ent in doc.ents:
            catch_ner_info(ent, i['context_filter'], show_detail)
            print(f'entity: {ent}, text: {ent.text}, NER label: {ent.label_}')
            print('context info:', scene_context)
          
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
    
    print('\n')
    print('-'*25, 'RESPONSE', '-'*25)
    print(f'${chatbot_name}: ${response}\n')
    