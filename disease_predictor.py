#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils import load_pickle_object, clean_str, cosine_similarity, load_nlp, get_counter
from config import (sent_data_file, feature_list, symptom_to_disease_file, disease_to_symptom_file,
                    model_encoder, categorical_nb_model, svm_model)

class DiseasePredictor():
    """This class will start the interactive disease predictor based on given symptoms.
    start() function starts the program and asks for the input.
    the output is a group suggestions based on the symptoms.
    """
    def __init__(self):
        """Load all generic objects.
        """
        self.nlp = load_nlp()
        self.feature_list_map = dict()
        self.features = []
        self.most_common_disease = set()
        self.closest_similar_disease = []
        # Load symptom to disease hash map
        self.symptom_to_disease_map = self.load_object(symptom_to_disease_file)
        # Load disease to symptoms hash map
        self.disease_to_symptom_map = self.load_object(disease_to_symptom_file)

    def load_feature_list(self):
        """Load the pickled file for retreiving the list of features used in the model training.py
        Each features here are symptom names.
        """
        # Unpickle the object from a binary file
        self.features = self.load_object(feature_list)
        # generate a feature name to index hash map.
        for i, feat in enumerate(self.features):
            self.feature_list_map[feat] = i

    def load_object(self, obj_name):
        """Load/Unpickle a python object binary file into a python object.

        Args:
            obj_name (string): filename of the stored binary pickle file.

        Returns:
            object : returns a python object.
        """
        return load_pickle_object(obj_name)

    def get_all_symptoms(self, disease_list: list) -> list():
        """Generates a hash map of disease -> list of symptoms key-value pairs

        Args:
            disease_list (liist): list of diseases

        Returns:
            list: a combined list of all symptoms for given disease in sorted order.
        """
        all_symptoms = set()
        for each_disease in disease_list:
            all_symptoms.update(self.disease_to_symptom_map.get(each_disease, set()))
        
        return sorted(all_symptoms)

    def get_most_probable_disease(self, disease_count_list: list) -> list:
        """Generate top most common disease out of a list of disease.

        Args:
            disease_count_list (list): a list of disease, sorted (descending) by its count with maximum 5 elements.

        Returns:
            list: a list of most_probable_disease upto 5 disease max. 
        """
        most_probable_disease = []
        max_count = 0
        for  disease, count in disease_count_list:
            if count >= max_count:
                max_count = count
                most_probable_disease.append(disease)

        return most_probable_disease


    def process_input_data(self, input_data: list, ouput_type: str = "sent"):
        """Generates possible, related symptoms for given input..

        Args:
            input_data (list): list of input symptoms in its root form.
            ouput_type (str, optional): "sent" or "token". Defaults to "sent".

        Returns:
            for output_type "sent" : return a combined string of all possible related symptoms.
        """
        combined_disease_list = []
        for each_symptom in input_data:
            combined_disease_list.append(self.symptom_to_disease_map.get(each_symptom, {}))

        disease_counter = get_counter(combined_disease_list)
        most_probable_disease = self.get_most_probable_disease(disease_counter.most_common(5))
        self.most_common_disease = set([name for name,count in disease_counter.most_common(3)])
        all_symptoms_list = self.get_all_symptoms(most_probable_disease)

        if ouput_type == "sent":
            return ' '.join(all_symptoms_list)
        
        return all_symptoms_list

    def init_models(self):
        """Initializes support varriables for all models
        """
        self.init_encoder_model()
        self.init_vectorization_model()

    def init_encoder_model(self):
        """Loads all supporting objects for prediction models.
        """
        self.label_encoder = self.load_object(model_encoder)
        self.load_feature_list()

    def get_disease_from_model(self, input_text: str, model, encoder: bool = False):
        """Use trained Model to predict disease with the processed symptoms.

        Args:
            input_text (str): a string line of all symptoms in its root form.
            model (ML model): ML model used for predicting the disease.
            encoder (bool, optional): to set label encoder to True or False. Defaults to False.
        """
        features_matrix = [0] * len(self.features)
        for symptom in input_text.split():
            if symptom in self.feature_list_map:
                features_matrix[self.feature_list_map.get(symptom)] = 1

        # Load features_matrix as a dataframe
        data = pd.DataFrame(features_matrix)
        # transpose the dataframe.
        data_frame = data.T
        # Add features as a column to the features_matrix
        data_frame.columns = self.features
        predicted_value = model.predict(data_frame)
        readable_value = predicted_value
        if encoder:
            # round of predicted value to nearest integer
            value = [round(predicted_value[0])]
            # decode the label encoded integer value to string
            readable_value = self.label_encoder.inverse_transform(value)

        predicted_disease = readable_value[0]
        return predicted_disease

            
    def init_vectorization_model(self):
        """Loads all supporting objects for vector calculations.
        """
        self.sentenced_input_file = pd.read_csv(sent_data_file)
        # get all the vectors of symptoms in training data
        with self.nlp.disable_pipes():
            vectors = np.array([self.nlp(data.symptom).vector for idx, data in self.sentenced_input_file.iterrows()])
        # calculate mean and center of combined vectors.
        self.vec_mean = vectors.mean(axis=0)
        self.centered = vectors - self.vec_mean

    def get_disease_from_vectorization_model(self, input_text: str):
        """Use cosine similarity calculations to identify closest symptoms to disease from given input.

        Args:
            input_text (str): a string line of all symptoms in its root form.
        """
        inp_vec = self.nlp(input_text).vector
        # calculate cosine angle difference of input string with all of training data.
        sims = np.array([cosine_similarity(inp_vec - self.vec_mean, vec) for vec in self.centered])
        # get the most closest vector
        most_similar = sims.argmax()
        # get the next 4 closest disease from vector
        self.closest_similar_disease = sims.argsort()[-5:-1][::-1]
        most_similar_disease = self.sentenced_input_file.iloc[most_similar].disease
        return most_similar_disease

    def pretty_print_predictions(self, predicted_values: list):
        """Print the final prediction results with suggestions and observations.

        Args:
            predicted_values (list): list of all predicted values along with its model name.
        """
        print("\n\n==========\n\n")
        print("The given symptoms closely matches with the following most common diseases:\n")

        for i, disease in enumerate(self.most_common_disease):
            print(f"{i+1}. {disease}")

        print("\n\nAccording to our 2 models, the closest matching diseases are")
        for i, (prediction, model_name) in enumerate(predicted_values):
            print(f"{i+1}. {model_name} model : {prediction}")

        print("\n\nThis symptoms can also be part of following diseases too:")

        for i, j in enumerate(self.closest_similar_disease):
            disease =  self.sentenced_input_file.iloc[j].disease
            print(f"{i+1}. {disease}")

        print("\n\nWe hope this information is useful for you.")
        print("\n\n==========\n\n")

    def run_prediction(self, input_string: str, method: str = "all"):
        """Cleans the input string of symptoms to its root form.
        Starts the prediction prrocess with the root symptom form.

        Args:
            input_string (str): a string line of all symptoms given as input.
            method (str, optional): defines the method of prediction to use: "vector", "model", or "all". Defaults to "all".
        """
        predicted_values = []
        # Define the ouput type for clean_str
        if method == "vector":
            output_type = "sent"
        else:
            output_type = "token"
        # clean the input data to its root form.
        input_text = clean_str(self.nlp(input_string.lower().strip()), typ=output_type)
        prediction_data = self.process_input_data(input_text)

        if method == "vector":
            # Evaluate cosine similarity of input vector with all diseases vectors
            prediction = self.get_disease_from_vectorization_model(prediction_data)
            predicted_values.append((prediction, "Vector"))
        elif method == "model":
            # Load Categorical Naive Bayes model and predict the disease from given symptoms.
            CNB_model = self.load_object(categorical_nb_model)
            prediction = self.get_disease_from_model(prediction_data, CNB_model)
            predicted_values.append((prediction, "Catg. Naive Bayes"))

            # Load SVC model and predict the disease from given symptoms.
            SVM_model = self.load_object(svm_model)
            prediction = self.get_disease_from_model(prediction_data, SVM_model, encoder=True)
            predicted_values.append((prediction, "SVM"))
        else:
            # Evaluate cosine similarity of input vector with all diseases vectors
            prediction = self.get_disease_from_vectorization_model(prediction_data)
            predicted_values.append((prediction, "Vector"))

            # Load Categorical Naive Bayes model and predict the disease from given symptoms.
            CNB_model = self.load_object(categorical_nb_model)
            prediction = self.get_disease_from_model(prediction_data, CNB_model)
            predicted_values.append((prediction, "Catg. Naive Bayes"))

            # Load SVC model and predict the disease from given symptoms.
            SVM_model = self.load_object(svm_model)
            prediction = self.get_disease_from_model(prediction_data, SVM_model, encoder=True)
            predicted_values.append((prediction, "SVM"))

        self.pretty_print_predictions(predicted_values)

 

    def start(self):
        """Starting point of the program.
        """
        self.init_models()
        while(1):
            input_str = input("Please enter 3 or more symptoms:\n")
            if input_str == "":
                print("please enter symptoms again")
                continue
            if input_str == "bye":
                print("THANK YOU !!!")
                break
            print("analyzing the symptoms.....")
            # Use all the available methods
            self.run_prediction(input_str)

            # Use only the Vectorization method
            # self.run_prediction(input_str, "vector")

            # Use only the ML model methods
            # self.run_prediction(input_str, "model")
            

if __name__ == '__main__':
    DiseasePredictor().start()