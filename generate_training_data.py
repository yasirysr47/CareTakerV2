#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import copy
import json
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from utils import load_nlp, pickle_object
from config import (disease_data_path, token_data_file, tfidf_data_file, sent_data_file,
                    symptom_counter_file, disease_to_symptom_file, symptom_to_disease_file)


class GenerateTrainingData():
    """Generates two CSV training data files along with two hash maps by using the processed raw data from DataStore."""
    def __init__(self):
        """Initialize all the generic objects."""
        self.nlp = load_nlp()
        self.stemmer = SnowballStemmer(language='english')
        self.word_count = {}
        self.token_dict = OrderedDict()
        self.symptom_to_disease_map = dict()
        self.disease_to_symptom_map = dict()

    def get_tfidf_vector(self, data_dict):
        """Generate TFIDF weights for all features"""
        disease_names = data_dict.keys()
        documents = [' '.join(doc) for doc in data_dict.values()]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        dense = vectors.todense()
        denselist = dense.tolist()
        feature_names = vectorizer.get_feature_names()
        dataframe = pd.DataFrame(denselist, columns=feature_names)
        dataframe.insert(0, 'disease', disease_names)
        return dataframe

    def save_feature_as_tfidf(self, data_map: dict):
        """Store the generated training features as TFIDF data in a csv file.

        Args:
            data_map (dict): hash map of title to list of symptoms (key: value) pairs

        Return:
            write the data into a CSV file.

        CSV file format :
        col 1 = disease 
        col 2...inf = each token (token = symptom root word)
        col 1 values are disease names
        other col values are tfidf frequency score
        each row is for one disease
        """
        #TFIDF weighted csv file
        vector_data = self.get_tfidf_vector(data_map)
        vector_data.to_csv(tfidf_data_file, encoding='utf-8', index=False)
    
    def save_feature_as_tokens(self, data_map: dict):
        """Store the generated training features as tokens (list of features) in a csv file.

        Args:
            data_map (dict): hash map of title to list of symptoms (key: value) pairs

        Return:
            write the data into a CSV file.

        CSV file format :
        col 1 = disease 
        col 2...inf = each token (token = symptom root word)
        col 1 values are disease names
        other col values are binary 0|1
        each row is for one disease
        """
        csv_file = open(token_data_file, "w+")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        token_obj = copy.copy(self.token_dict)
        symptom_list = token_obj.keys()
        # First column of csv is disease name.
        header = ['disease']
        # Adding all symptoms as columns.
        header.extend(symptom_list)
        csv_writer.writerow(header)
        # Write all disease title and its corrresponding binary symptoms list in each rows.
        for title, tokens in data_map.items():
            if len(tokens) < 2:
                continue
            token_obj = copy.copy(self.token_dict)
            row = [title]
            for token in tokens:
                token_obj[token] = 1
            
            row.extend(token_obj.values())
            csv_writer.writerow(row)

    def save_feature_as_sents(self, data_map: dict):
        """Store the generated training features as a line of string (string of symptoms) in a csv file.

        Args:
            data_map (dict): hash map of title to string of symptoms (key: value) pairs

        Return:
            write the data into a CSV file.

        CSV file format :
        col 1 = disease 
        col 2 = symptoms
        col1 value = disease name 
        col2 value = one sentence of symptoms for one disease
        each row is for one disease
        """
        csv_file = open(sent_data_file, "w+")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # Only two defined columns are there, disease and symptom.
        csv_writer.writerow(['disease', 'symptom'])
        # Write all disease title and its corresponding symptom string sentence in each rows.
        for title, words in data_map.items():
            if len(words) < 2:
                continue
            sent = ' '.join(sorted(list(words)))
            csv_writer.writerow([title, sent])
        
    def tokenize_sent(self, title: str, sent: str):
        """Tokenize the string of symptoms into its root form.

        Args:
            title (string): name of disease
            sent (string): a string sentence of all its symptoms
        """
        token_set = set()
        doc = self.nlp(sent)
        for word in doc:
            # Ignore all the punctuations, stop words, non-alphabetic characters, white spaces and words with 2 or less characters.
            if (not word.is_punct and not word.is_stop
            and not word.is_space and word.is_alpha
            and len(word) > 2):
                # Get the stem/root word.
                stem_word = self.stemmer.stem(word.text)
                token_set.add(stem_word)
                # TODO: To ignore stem_word that are not valid symptoms or is a generic word.
                # Generate a symptom to disease hash map. And keep track of the count of symptoms.
                if stem_word in self.word_count:
                    self.word_count[stem_word] += 1
                    self.symptom_to_disease_map[stem_word].add(title)
                else:
                    self.token_dict[stem_word] = 0
                    self.word_count[stem_word] = 1
                    self.symptom_to_disease_map[stem_word] = {title}
        # Generate a disease to symptom hash map.
        if not self.disease_to_symptom_map.get(title):
            self.disease_to_symptom_map[title] = token_set
        else:
            self.disease_to_symptom_map[title].update(token_set)

    def dump_all_token(self):
        """Dump all symptom counts to a JSON file."""
        # Sort the symptom count dict from most highest occuring symptom to least occuring one.
        self.word_count = {k:v for k, v in sorted(self.word_count.items(), key=lambda itm: itm[1], reverse=True)}
        with open(symptom_counter_file, "w+") as fp:
            json.dump(self.word_count, fp, indent=4)

    def generate_data_and_features(self):
        """Generate the training CSV files and hash maps from all the disease files available at DataStore."""
        syptom_sent_set = set()
        flag = 0
        title_flag = 0
        limit = 0
        title = ''
        # Get all disease detail filenames from DataStore.
        files = [f for f in listdir(disease_data_path) if isfile(join(disease_data_path, f))]
        for fil in files:
            title = ''
            title_flag = 0
            if limit == 2000:
                break
            limit += 1
            # Open each disease file name for processing.
            fp = open(os.path.join(disease_data_path, fil), "r")
            line_no = 0
            for each_line in fp.readlines():
                each_line = each_line.strip().lower()
                if not title_flag and not title and not line_no:
                    title = each_line
                    title_flag = 1
                    #TODO: To ignore titles that are not valid disease names.

                # Starting of the symptoms section in the file.
                if not flag and each_line == 'symptoms':
                    flag = 1
                elif not flag:
                    continue
                # Each symptom in the file starts with the arrow sign.
                if flag and each_line.startswith('->'):
                    sent = each_line.strip('->')
                    syptom_sent_set.add(sent)
                    self.tokenize_sent(title, sent)
                if flag and each_line.startswith('---'):
                    flag = 0
                    break
                line_no += 1

        # Save the disease and synptom data into CSV files.
        self.save_feature_as_tfidf(self.disease_to_symptom_map)
        self.save_feature_as_sents(self.disease_to_symptom_map)
        self.save_feature_as_tokens(self.disease_to_symptom_map)
        self.dump_all_token()
        # Save disease to symptom and symptom to disease hash maps into a pickle file.
        pickle_object(self.disease_to_symptom_map, disease_to_symptom_file)
        pickle_object(self.symptom_to_disease_map, symptom_to_disease_file)



if __name__ == '__main__':
    datagen = GenerateTrainingData()
    datagen.generate_data_and_features()