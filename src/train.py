import ast
import json
import os
import pandas as pd
import numpy as np

from models.train import train_chatbot_model, train_ner_model
from data.generate_data import generate_ner_data, generate_chatbot_data, get_intents
from utils import PROJECT_ROOT_PATH


def train_diagnose_model(output_name: str, intents: list):
    print('-' * 25 + ' start of train model:', output_name + ' ' + '-'*25)
    generate_chatbot_data(False, name=output_name, intents=intents)

    df = pd.read_csv(os.path.join(
        PROJECT_ROOT_PATH, f'data/{output_name}.csv'))
    # pare string list to numeric list
    df['pattern'] = df['pattern'].transform(lambda x: ast.literal_eval(x))
    df['label'] = df['label'].transform(lambda x: ast.literal_eval(x))

    X = df['pattern'].tolist()
    y = df['label'].tolist()
    X = np.asanyarray(X).astype(np.int32)
    y = np.asanyarray(y).astype(np.int32)

    train_chatbot_model(X, y, name=output_name)
    print('-' * 25 + ' end of train model:', output_name + ' ' + '-'*25)


if __name__ == '__main__':

    # generate_ner_data()
    # train_ner_model(True)

    # sub model
    train_diagnose_model(output_name='form',
                         intents=json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/eform/form-expert.json')).read()))

    train_diagnose_model(output_name='upload',
                         intents=json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/eform/form-upload-expert.json')).read()))

    train_diagnose_model(output_name='book-flight',
                         intents=json.loads(open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/booking/booking-expert.json')).read()))

    # main model
    train_diagnose_model(output_name='chatbot', intents=get_intents())

    # # excute spacy train ner script
    # subprocess.call(['sh', os.path.join(PROJECT_ROOT_PATH, 'src/models/ner/spacy-train.sh')])
