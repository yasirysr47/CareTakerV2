import pickle
import spacy
import collections
import functools
import operator
import numpy as np
from nltk.stem.snowball import SnowballStemmer



STEMMER = SnowballStemmer(language='english')

def load_nlp():
    nlp = spacy.load('en_core_web_lg')
    return nlp

def pickle_object(model, filename):
    with open(filename, 'wb') as fp:  
        pickle.dump(model, fp)

def load_pickle_object(filename):
    obj = None
    with open(filename, 'rb') as fp:  
        obj = pickle.load(fp)
    return obj

def clean_str(doc, typ = "sent"):
    token_set = set()
    for word in doc:
        if (not word.is_punct and not word.is_stop
            and not word.is_space and word.is_alpha
            and len(word) > 2):
            stem_word = STEMMER.stem(word.text)
            token_set.add(stem_word)

    if typ == "token":
        return sorted(token_set)
    sent = ' '.join(sorted(token_set))
    return sent


def cosine_similarity(vector_a, vector_b):
    return (np.dot(vector_a, vector_b) / np.sqrt(vector_a.dot(vector_a) * vector_b.dot(vector_b)))

def get_counter(list_of_symptoms):
    counter = functools.reduce(
        operator.add,
        (
        collections.Counter(set(symptoms)) for symptoms in list_of_symptoms
        )
    )
    return counter