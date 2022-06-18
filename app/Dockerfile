FROM python:3.8-slim-buster

WORKDIR /chatbot-app

COPY requirments.txt requirments.txt
RUN pip install -r requirments.txt
RUN python -m spacy download en_core_web_sm

COPY . /chatbot-app

CMD [ "python", "./src/main.py" ]