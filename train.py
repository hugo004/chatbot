from model import Model
from data import get_data

def run():
  train_data = get_data()
  model = Model()
  model.fit(X_train=train_data["x_train"], y_train=train_data['y_train'])


run()