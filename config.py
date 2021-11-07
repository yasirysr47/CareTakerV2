# -*- coding: utf-8 -*-
import os
import sys
# depended upon data_store for retrieving data
sys.path.insert(0, '/Users/myasir/Personal')
from DataStore.dir import Dir

PATH = Dir('..')
disease_data_path = PATH.disease_data_dir
# get dependent paths
models_path = os.path.join('./', "models")
ml_datagen_path = os.path.join('./', "ml_datagen")

# training data files
token_data_file = os.path.join(ml_datagen_path, "feature_as_tokens_new.csv")
sent_data_file = os.path.join(ml_datagen_path, "feature_as_sents_new.csv")
symptom_counter_file = os.path.join(ml_datagen_path, "symptoms_count.txt")

# location for pickled models and objects
gaussian_nb_model = os.path.join(models_path, "gaussian_nb_model.pkl")
random_forest_model = os.path.join(models_path, "random_forest_model.pkl")
regressor_model = os.path.join(models_path, "regressor_model.pkl")
knn_model = os.path.join(models_path, "knn_model.pkl")
svc_model = os.path.join(models_path, "svc_model.pkl")
bernouli_nb_model = os.path.join(models_path, "bernouli_nb_model.pkl")
categorical_nb_model = os.path.join(models_path, "categorical_nb_model.pkl")
svm_model = os.path.join(models_path, "svm_model.pkl")
linear_svm_model = os.path.join(models_path, "linear_svm_model.pkl")
svr_model = os.path.join(models_path, "svr_model.pkl")
linear_svr_model = os.path.join(models_path, "linear_svr_model.pkl")

model_encoder = os.path.join(models_path, "model_encoder.pkl")
feature_list = os.path.join(models_path, "feature_list.pkl")
model_report_file = os.path.join(models_path, "model_report.txt")

disease_to_symptom_file = os.path.join(ml_datagen_path, "disease_to_symptom_map.pkl")
symptom_to_disease_file = os.path.join(ml_datagen_path, "symptom_to_disease_map.pkl")