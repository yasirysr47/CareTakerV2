import os
import sys
#depended upon scrapy and genie and data_store
sys.path.insert(0, '/Users/myasir/Personal')
from DataStore.dir import Dir

PATH = Dir('..')
disease_data_path = PATH.disease_data_dir

models_path = os.path.join('./', "models")
ml_datagen_path = os.path.join('./', "ml_datagen")
token_data_file = os.path.join(ml_datagen_path, "feature_as_tokens_new.csv")
sent_data_file = os.path.join(ml_datagen_path, "feature_as_sents_new.csv")
symptom_counter_file = os.path.join(ml_datagen_path, "symptoms_count.txt")


naive_bayes_model = os.path.join(models_path, "naive_bayes_model.pkl")
random_forest_model = os.path.join(models_path, "random_forest_model.pkl")
model_encoder = os.path.join(models_path, "model_encoder.pkl")
feature_list = os.path.join(models_path, "feature_list.pkl")
model_report_file = os.path.join(models_path, "model_report.txt")

disease_to_symptom_file = os.path.join(ml_datagen_path, "disease_to_symptom_map.pkl")
symptom_to_disease_file = os.path.join(ml_datagen_path, "symptom_to_disease_map.pkl")