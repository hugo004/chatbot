
import os
import spacy

from src.utils import PROJECT_ROOT_PATH


nlp = spacy.load(os.path.join(PROJECT_ROOT_PATH, 'models/ner/model-best'))
def test_ner(sentence):
  doc = nlp(sentence.lower())
  for ent in doc.ents:
    print(ent, ent.label_, ent.text)
    
    
test_ner('To Japan')