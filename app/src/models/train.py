import math
import os
import logging
import spacy
import json

from utils import PROJECT_ROOT_PATH, ROOT_PATH
from spacy.tokens import DocBin
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import gradient_descent_v2
from spacy.util import filter_spans

logging.basicConfig(level=logging.INFO)


def train_ner_model(show_detail=False):
    '''
    train custom NER
    '''
    train_data = json.loads(
        open(os.path.join(PROJECT_ROOT_PATH, 'data/ner-train-data.json')).read())
    nlp = spacy.blank('en')

    db = DocBin()
    for data in train_data['texts']:
        text = data['text'].lower()
        entities = data['entities']

        doc = nlp.make_doc(text)
        if show_detail:
            logging.info(f'training -> text: {text} entities: {entities}')

        ents = []
        for ent in entities:
            start, end, label = ent['start'], ent['end'], ent['label']

            span = doc.char_span(start, end, label=label.upper())
            if span is None:
                logging.warn('skipping entity')
            else:
                ents.append(span)
                if show_detail:
                    logging.info(f'append ->  {span}')

        # handle long span overlap
        doc.ents = filter_spans(ents)
        db.add(doc)

    db.to_disk(os.path.join(ROOT_PATH, 'models/ner/train.spacy'))
    logging.info('NER model saved')


def train_chatbot_model(X_train, y_train, name='chatbot'):
    '''
    train chatbot model
    '''
    factor = 0.1
    input_unit = X_train[0].shape[0]
    output_unit = len(y_train[0])
    data_sampels = len(X_train[0])
    hidden_unit = math.floor(
        data_sampels / ((input_unit + output_unit) * factor))

    model = Sequential()
    model.add(Dense(input_unit, input_shape=(
        len(X_train[0]),), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_unit, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(y_train[0]), activation='softmax'))

    SGD = gradient_descent_v2.SGD
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        min_delta=0.1
    )

    hist = model.fit(X_train,
                     y_train,
                     epochs=200,
                     batch_size=5,
                     verbose=1,
                     callbacks=[early_stop])
    model.save(os.path.join(PROJECT_ROOT_PATH, f'models/{name}.h5'), hist)

    logging.info(f'{name} model saved')
