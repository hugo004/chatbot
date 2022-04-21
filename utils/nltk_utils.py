import nltk
# run at first time
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def lematize(word: str):
    return lemmatizer.lemmatize(word=word.lower())
  
def tokenize(word: str):
    return nltk.word_tokenize(word)

def stem(word: str):
    return stemmer.stem(word=word.lower())
