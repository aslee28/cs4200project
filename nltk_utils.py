import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

def stemming(word):
    """
    stemming = find the root form of the word
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    """
    sentence_words = [stemming(word) for word in tokenized_sentence]
    # create a new bag array and initialize all values to 0.
    bagOfWords = np.zeros(len(words), dtype=np.float32)
    for index, word in enumerate(words):
        if word in sentence_words:
            bagOfWords[index] = 1
    return bagOfWords