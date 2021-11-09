#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
import spacy
import functools
import operator
import numpy as np
from collections import Counter
from nltk.stem.snowball import SnowballStemmer


# nltk snowballstemmer for getting stem/root word of any given word
STEMMER = SnowballStemmer(language='english')

def load_nlp():
    """Load english nlp models for spacy

    Returns:
        Spacy object: spacy object with english model loaded
    """
    nlp = spacy.load('en_core_web_lg')
    return nlp

def pickle_object(object, filename: str):
    """Write any python object into a binary file.

    Args:
        object: any python object
        filename (string): filename to store the object with 'pkl' extension (<filename>.pkl)
    """
    with open(filename, 'wb') as fp:  
        pickle.dump(object, fp)

def load_pickle_object(filename: str):
    """Load/Unpickle a python object binary file into a python object.

    Args:
        filename (string): pickle filename to be loaded. (<filename>.pkl)

    Returns:
        python object: returns the loaded file as python object of any type
    """
    obj = None
    with open(filename, 'rb') as fp:  
        obj = pickle.load(fp)
    return obj

def clean_str(doc, typ = "sent"):
    """Convert the given spacy string object into its stem/root form.

    Args:
        doc (spacy object): spacy string object
        typ (string): choose the return type as "token" or "sent" (default)

    Returns:
        string or list: returns string by default unless specifies in the typ param.
    Eg:
    typ = "sent" : 'fever, pain, ach, muscl' -> string
    typ = "token" : [fever, pain, ach, muscl] -> list
    """
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
    """Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.
    It is defined to equal the cosine of the angle between them,
    which is also the same as the inner product of the same vectors normalized to both have length 1.
    Values range between -1 and 1, where -1 is perfectly dissimilar and 1 is perfectly similar.

    Args:
        vector_a (vector): vector of a string
        vector_b (vector): vector of a string

    Returns:
        float: float value between -1 tp 1
    """
    return (np.dot(vector_a, vector_b) / np.sqrt(vector_a.dot(vector_a) * vector_b.dot(vector_b)))

def get_counter(list_of_symptoms: list) -> list:
    """Generates a counter of words with its number of occurences

    Args:
        list_of_symptoms (list): A list of set of symptoms -> [{...},{...},{..},..]

    Returns:
        Counter object: counter object with word and its occurence number
    """
    counter = functools.reduce(
        operator.add,
        (
        Counter(set(symptoms)) for symptoms in list_of_symptoms
        )
    )
    return counter
