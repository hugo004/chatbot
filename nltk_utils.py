import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def lematize(word: str):
    return lemmatizer.lemmatize(word=word.lower())
  

def tokenize(word: str):
    return nltk.word_tokenize(word)