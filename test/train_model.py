from ast import Lambda, literal_eval
import ast
import os
import pandas as pd
import numpy as np
import subprocess

from src.models.train import train_chatbot_model, train_ner_model
from src.data.generate_data import generate_ner_data, generate_chatbot_data
from src.utils import PROJECT_ROOT_PATH


if __name__ == '__main__':  
  
  # generate_ner_data()
  # train_ner_model(True)
  
  generate_chatbot_data(False)
  
  df =  pd.read_csv(os.path.join(PROJECT_ROOT_PATH, 'data/chatbot-intents.csv'))
  # pare string list to numeric list
  df['pattern'] = df['pattern'].transform(lambda x: ast.literal_eval(x))
  df['label'] = df['label'].transform(lambda x: ast.literal_eval(x))
  
  X = df['pattern'].tolist()
  y = df['label'].tolist()
  X = np.asanyarray(X).astype(np.int32)
  y = np.asanyarray(y).astype(np.int32)
  
  train_chatbot_model(X, y)
  

  
  # # excute spacy train ner script
  # subprocess.call(['sh', os.path.join(PROJECT_ROOT_PATH, 'src/models/ner/spacy-train.sh')])