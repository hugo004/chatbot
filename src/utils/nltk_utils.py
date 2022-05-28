import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from typing import List, Union

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# run at first time
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')


def lematize(word: str):
    return lemmatizer.lemmatize(word=word.lower())


def tokenize(word: str):
    return nltk.word_tokenize(word)


def stem(word: str):
    return stemmer.stem(word=word.lower())


def bow(tokenized_sentence: Union[List[str], str], words: List[str]):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
