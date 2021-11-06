import pandas as pd
import numpy as np
from utils import load_pickle_object, clean_str, cosine_similarity, load_nlp, get_counter
from config import (naive_bayes_model, feature_list, model_encoder, random_forest_model,
                    sent_data_file, disease_to_symptom_file, symptom_to_disease_file)



class DiseasePredictor():
    def __init__(self):
        self.nlp = load_nlp()
        self.feature_list_map = dict()
        self.features = []
        self.features_matrix = []
        self.most_probable_disease = set()
        self.most_similar_disease = None
        self.predicted_disease = None
        self.closest_similar_disease = []
        self.naive_bayes_model = self.load_object(naive_bayes_model)
        self.vec_file = pd.read_csv(sent_data_file)
        # self.random_forest_model = self.load_object(random_forest_model)
        self.label_encoder = self.load_object(model_encoder)
        self.symptom_to_disease_map = self.load_object(symptom_to_disease_file)
        self.disease_to_symptom_map = self.load_object(disease_to_symptom_file)

        pass

    def load_feature_list(self):
        self.features = load_pickle_object(feature_list)

        for i, feat in enumerate(self.features):
            self.feature_list_map[feat] = i


    def load_object(self, obj_name):
        return load_pickle_object(obj_name)

    def get_all_symptoms(self, disease_list) -> list():
        all_symptoms = set()
        for each_disease in disease_list:
            all_symptoms.update(self.disease_to_symptom_map.get(each_disease, set()))
        
        return sorted(all_symptoms)

    def process_input_data(self, input_data, input_type):
        if input_type == "sent":
            input_list = input_data.split()
        else:
            input_list = input_data
        
        combined_disease_list = []
        for each_symptom in input_list:
            combined_disease_list.append(self.symptom_to_disease_map.get(each_symptom, {}))

        disease_counter = get_counter(combined_disease_list)
        print(disease_counter)
        self.most_probable_disease = set([name for name,count in disease_counter.most_common(3)])
        all_symptoms_list = self.get_all_symptoms(self.most_probable_disease)

        if input_type == "sent":
            return ' '.join(all_symptoms_list)
        
        return all_symptoms_list

    def init_naive_bayes_model(self):
        self.load_feature_list()

    def predict_by_naive_bayes_model(self, input_text):
        self.features_matrix = [0] * len(self.features)
        for symptom in input_text:
            if symptom in self.feature_list_map:
                self.features_matrix[self.feature_list_map.get(symptom)] = 1

        data = pd.DataFrame(self.features_matrix)
        data_frame = data.T
        predicted_values1 = self.naive_bayes_model.predict(data_frame)
        # print("possible disease predicted are 1:")
        readable_values = self.label_encoder.inverse_transform(predicted_values1)
        # print(readable_values)
        self.predicted_disease = readable_values[0]
        # print("Your symptoms are very similar to {}".format(readable_values[0]))
        
        del self.features_matrix
        del readable_values
            
    def init_vectorization_model(self):
        with self.nlp.disable_pipes():
            vectors = np.array([self.nlp(data.symptom).vector for idx, data in self.vec_file.iterrows()])

        self.vec_mean = vectors.mean(axis=0)
        self.centered = vectors - self.vec_mean

    def predict_by_vectorization_model(self, input_text):
        inp_vec = self.nlp(input_text).vector
        sims = np.array([cosine_similarity(inp_vec - self.vec_mean, vec) for vec in self.centered])
        

        most_similar = sims.argmax()
        self.closest_similar_disease = sims.argsort()[-5:-1][::-1]
        # import pdb; pdb.set_trace()
        self.most_similar_disease = self.vec_file.iloc[most_similar].disease
        # print("1. Your symptoms are very similar to {}".format(ms_dis))
        # print("and these are the other possibilities in decreasing chance of occurence")
        # for i, j in enumerate(self.closest_similar_disease):
        #     ns_dis =  self.vec_file.iloc[j].disease
        #     print("{}. {}".format(i+2, ns_dis))

    def pretty_print_predictions(self):
        print("\n\n==========\n\n")
        print("The given symptoms closly matches with the following most common diseases:\n")
        for i, disease in enumerate(self.most_probable_disease):
            print(f"{i+1}. {disease}")

        print("\n\naccording to our 2 models, the closest matching diseases are")
        print("1 => vectorization model : {}".format(self.most_similar_disease))
        print("2 => naive bayes model : {}".format(self.predicted_disease))

        print("\n\nthis symptoms can also be part following diseases too:")

        for i, j in enumerate(self.closest_similar_disease):
            disease =  self.vec_file.iloc[j].disease
            print(f"{i+1}. {disease}")

        print("\n\nWe hope this information is useful for you.")

    def run_prediction(self, input_string, method="vector"):
        if method == "vector":
            self.init_vectorization_model()
            input_type = "sent"
        else:
            self.init_naive_bayes_model()
            input_type = "token"
        input_text = clean_str(self.nlp(input_string.lower().strip()), typ=input_type)
        # print(input_text)
        prediction_data = self.process_input_data(input_text, input_type)
        
        if method == "vector":
            self.predict_by_vectorization_model(prediction_data)
        else:
            self.predict_by_naive_bayes_model(prediction_data)

 

    def start(self):
        while(1):
            input_str = input("enter symptoms\n")
            if input_str == "bye":
                break
            print("analyzing the symptoms.....")
            self.run_prediction(input_str, "vector")
            self.run_prediction(input_str, "naive_bayes")
            self.pretty_print_predictions()
            print("\n\n==========\n\n")
            

if __name__ == '__main__':
    DiseasePredictor().start()
    # DiseasePredictor().run_prediction("vector")
    # DiseasePredictor().run_prediction("naive_bayes")
    # fever, loss of strength, fluid bump