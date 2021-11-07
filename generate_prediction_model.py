#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils import pickle_object, load_nlp
from config import (token_data_file, svm_model, feature_list, categorical_nb_model, model_encoder)


class TrainModel():
    """Train ML models to use for disease predictions.
    """
    def __init__(self):
        """Loads training data files and spacy object with english langugae model.
        """
        self.nlp = load_nlp()
        self.tokenized_input_file = pd.read_csv(token_data_file)

    def save_object(self, object, filename: str):
        """Save a python object to a binary file for later use.

        Args:
            object (python object): any type of python object.
            filename (str): filename to save the python object.
        """
        pickle_object(object, filename)

    def train_and_save_model(self):
        """Train the ML model and store it in a binary file.
        """
        label_encoder = LabelEncoder()
        # Y is the target values w.r.t training data.
        Y = self.tokenized_input_file.disease
        # features are all columns except the target column (disease).
        features = self.tokenized_input_file.columns.drop('disease')
        self.save_object(features, feature_list)
        # X is the training data.
        X = self.tokenized_input_file[features]
        # Get some or all cases for testing purposes.
        _, default_X_test, _, default_y_test = train_test_split(X, Y, test_size=0.99, random_state=5)
        
        # encoding string into integers.
        encoded_Y = label_encoder.fit_transform(Y)
        _, encoded_X_test, _, encoded_y_test = train_test_split(X, encoded_Y, test_size=0.99, random_state=5)

        # SVM model.
        SVM_model = SVC()
        # Naive Bayes model
        CNB_model = CategoricalNB()

        models = [SVM_model, CNB_model]
        # start trainng the models
        for i, model in enumerate(models):
            start_time = time.time()
            i += 1
            print("Training {} started....".format(model))
            X_train = X
            if model == SVM_model:
                # use label encoded values for svm
                y_train = encoded_Y
                X_test = encoded_X_test
                y_test = encoded_y_test
            else:
                y_train = Y
                X_test = default_X_test
                y_test = default_y_test
            # actual training starts here.
            model.fit(X_train,y_train)
            
            print("Training completed !!!!")
            # evaluate the model accuracy.
            score = model.score(X_test, y_test)*100
            print("Model test accuracy: {:.3f} %".format(score))
            end_time = time.time()
            print("Total time taken for training: {} s".format(end_time-start_time))
        
        # Save the models and encoder into a file. (pickling)
        self.save_object(SVM_model, svm_model)
        self.save_object(CNB_model, categorical_nb_model)
        self.save_object(label_encoder, model_encoder)


if __name__ == '__main__':
    TrainModel().train_and_save_model()