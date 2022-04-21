import spacy

def test_ner(text: str):
  nlp = spacy.load('./models/ner/model-best')
  doc = nlp(text)
  for ent in doc.ents:
    print(ent.text, ent.label_)

test_ner("from Hong Kong to Japan")