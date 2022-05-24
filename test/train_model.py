from ast import Lambda, literal_eval
import ast
import json
import os
import pandas as pd
import numpy as np
import subprocess

from src.models.train import train_chatbot_model, train_ner_model
from src.data.generate_data import generate_ner_data, generate_chatbot_data, get_intents
from src.utils import PROJECT_ROOT_PATH


def train_diagnose_model(output_name: str, intents: list):
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


if __name__ == '__main__':

    # generate_ner_data()
    # train_ner_model(True)

    # sub model
    train_diagnose_model(output_name='form', intents=json.loads(
        open(os.path.join(PROJECT_ROOT_PATH, 'data/intents/eform/custom-intents.json')).read()))

    # main model
    train_diagnose_model(output_name='chatbot', intents=get_intents())

    # # excute spacy train ner script
    # subprocess.call(['sh', os.path.join(PROJECT_ROOT_PATH, 'src/models/ner/spacy-train.sh')])
