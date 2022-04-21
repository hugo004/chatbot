import sys
import spacy
import json
import os
from spacy.tokens import DocBin
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL

def ner_train_data():
  train_data = json.loads(open('./data/ner-train.json').read())
  nlp = spacy.blank('en')
  config = {"model": DEFAULT_TAGGER_MODEL}
  nlp.add_pipe("tagger", config=config)
  
  db = DocBin()
  for data in train_data['texts']:
    text = data['text']
    entities = data['entities']
    print('text:', text, 'ano:', entities)
    doc = nlp.make_doc(text)
    print('doc:', doc)
    ents = []
    for ent in entities:
      start, end, label  = ent['start'], ent['end'], ent['label']
      
      span = doc.char_span(start, end, label=label.upper())
      print('span:', span)
      if span is None:
        print('skipping entity')
      else:
        ents.append(span)
    doc.ents = ents
    db.add(doc)
  db.to_disk('./models/ner/train.spacy')
  print('spacy model saved')


ner_train_data()
