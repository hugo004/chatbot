import json
import os
import numpy as np
import spacy
import pickle

from keras.models import load_model
from typing import List, Union
from utils import PROJECT_ROOT_PATH
from utils.nltk_utils import tokenize, bow


ner_nlp = spacy.load(os.path.join(
    PROJECT_ROOT_PATH, 'models/ner/model-extend'))

threshold = 0.6


def preprocess_sentence(sentence: Union[List[str], str], words):
    # tokenized
    tokenized_sentence = tokenize(sentence)
    bag = bow(tokenized_sentence, words)

    return bag


def predict_intent(sentence: str, labels, words, model):
    sentence = preprocess_sentence(sentence, words=words)
    results = model.predict(np.array([sentence]))[0]
    results = [[i, r] for i, r in enumerate(results) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    intent_list = []
    for result in results:
        intent_list.append({
            'intent': labels[result[0]],
            'probability': result[1]
        })

    return intent_list


def predict_ner(sentence: str):
    doc = ner_nlp(sentence)
    return doc
