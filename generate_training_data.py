import os
import sys
import copy
import json
import csv


from os import listdir
from os.path import isfile, join
from collections import OrderedDict
from nltk.stem.snowball import SnowballStemmer
from utils import load_nlp, pickle_object
from config import (disease_data_path, token_data_file, sent_data_file, symptom_counter_file,
disease_to_symptom_file, symptom_to_disease_file)




class GenerateTrainingData():
    def __init__(self):
        self.nlp = load_nlp()
        self.stemmer = SnowballStemmer(language='english')
        self.word_count = {}
        self.token_dict = OrderedDict()
        self.symptom_to_disease_map = dict()
        self.disease_to_symptom_map = dict()

    def save_feature_as_tokens(self, data_set):
        '''
        data_set = (title, (lot of tokens))
        excel file format :
        col 1 = disease 
        col 2...inf = each token
        col1 value = disease name
        other col value is binary 0|1
        each row is for each set of tokens
        '''
        csv_file = open(token_data_file, "w+")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        token_obj = copy.copy(self.token_dict)
        symptom_list = token_obj.keys()
        header = ['disease']
        header.extend(symptom_list)
        csv_writer.writerow(header)
        for title, tokens in data_set.items():
            if len(tokens) < 2:
                continue
            token_obj = copy.copy(self.token_dict)
            row = [title]
            for token in tokens:
                token_obj[token] = 1
            
            row.extend(token_obj.values())
            csv_writer.writerow(row)

    def save_feature_as_sents(self,data_list):
        '''
        data_set = (title, sentence)
        excel file format :
        col 1 = disease 
        col 2 = symptoms
        col1 value = disease name 
        col2 value = one sentence of one disease
        each row is for one disease
        '''
        csv_file = open(sent_data_file, "w+")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['disease', 'symptom'])
        for title, words in data_list.items():
            if len(words) < 2:
                continue
            sent = ' '.join(sorted(list(words)))
            csv_writer.writerow([title, sent])
        
    def tokenize_sent(self, title, sent):
        token_set = set()
        doc = self.nlp(sent)
        for word in doc:
            if (not word.is_punct and not word.is_stop
            and not word.is_space and word.is_alpha
            and len(word) > 2):
                stem_word = self.stemmer.stem(word.text)
                token_set.add(stem_word)

                if stem_word in self.word_count:
                    self.word_count[stem_word] += 1
                    self.symptom_to_disease_map[stem_word].add(title)
                else:
                    self.token_dict[stem_word] = 0
                    self.word_count[stem_word] = 1
                    self.symptom_to_disease_map[stem_word] = {title}
            
                '''
                token_set.add(word.lemma_)
                if word.lemma_ in self.word_count:
                    self.word_count[word.lemma_] += 1
                else:
                    self.token_dict[word.lemma_] = 0
                    self.word_count[word.lemma_] = 1
                '''
        if not self.disease_to_symptom_map.get(title):
            self.disease_to_symptom_map[title] = token_set
        else:
            self.disease_to_symptom_map[title].update(token_set)

    def get_all_token(self):
        self.word_count = {k:v for k, v in sorted(self.word_count.items(), key=lambda itm: itm[1], reverse=True)}
        with open(symptom_counter_file, "w+") as fp:
            json.dump(self.word_count, fp, indent=4)

    def generate_data_and_features(self):
        syptom_sent_set = set()
        flag = 0
        title_flag = 0
        limit = 0
        title = ''

        files = [f for f in listdir(disease_data_path) if isfile(join(disease_data_path, f))]
        for fil in files:
            title = ''
            title_flag = 0
            if limit == 2000:
                break
            limit += 1
            fp = open(os.path.join(disease_data_path, fil), "r")
            line_no = 0
            for each_line in fp.readlines():
                each_line = each_line.strip().lower()
                if not title_flag and not title and not line_no:
                    title = each_line
                    title_flag = 1
                    #TODO: to fix data title issue
                    # if title.startswith(('y', 'eart')) and not title.startswith('yeast'):
                    #     title = "h{}".format(title)
                    
                if not flag and each_line == 'symptoms':
                    flag = 1
                elif not flag:
                    continue
                if flag and each_line.startswith('->'):
                    sent = each_line.strip('->')
                    syptom_sent_set.add(sent)
                    self.tokenize_sent(title, sent)
                if flag and each_line.startswith('---'):
                    flag = 0
                    break
                line_no += 1


        self.save_feature_as_sents(self.disease_to_symptom_map)
        self.save_feature_as_tokens(self.disease_to_symptom_map)
        self.get_all_token()
        # import pdb; pdb.set_trace()
        pickle_object(self.disease_to_symptom_map, disease_to_symptom_file)
        pickle_object(self.symptom_to_disease_map, symptom_to_disease_file)



if __name__ == '__main__':
    datagen = GenerateTrainingData()
    datagen.generate_data_and_features()