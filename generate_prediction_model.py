import time
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils import pickle_object, clean_str, load_nlp
from config import (token_data_file, sent_data_file, naive_bayes_model, random_forest_model,
                    model_encoder, feature_list, model_report_file)


class GenerateModel():
    def __init__(self):
        self.nlp = load_nlp()
        self.vec_file = pd.read_csv(sent_data_file)
        self.rf_file = pd.read_csv(token_data_file)

    def clean_str(self, st, typ = "sent"):
        doc = self.nlp(st)
        token_set = set()
        for word in doc:
            if (not word.is_punct and not word.is_stop
            and not word.is_space and word.is_alpha):
                token_set.add(word.lemma_)

        if typ == "token":
            return sorted(list(token_set))
        sent = ' '.join(sorted(list(token_set)))
        return sent

    def save_object(self, object, filename):
        pickle_object(object, filename)

    def save_model_report(self, info):
        with open(model_report_file, "w+") as fp:
            fp.write("\n".join(info))        

    def get_best_model(self):
        Y = self.rf_file.disease
        model_report = []
        #word_map_key, word_map_val, Y = self.symptom_map(Y)
        features = self.rf_file.columns.drop('disease')
        pickle_object(features, feature_list)
        X = self.rf_file[features]
        orig_X_train, orig_X_test, orig_y_train, orig_y_test = train_test_split(X, Y, test_size=0.000001, random_state=1)
        xx_train, orig_Z_test, yy_train, orig_z_test = train_test_split(X, Y, test_size=0.99, random_state=1)
        # import pdb; pdb.set_trace()
        
        scaler = StandardScaler()
        scaler.fit(orig_X_train)

        orig_X_train = scaler.transform(orig_X_train)
        orig_X_test = scaler.transform(orig_X_test)

        
        model1 = RandomForestRegressor(n_estimators=200, random_state=1)
        model2 = XGBRegressor(n_estimators=250, learning_rate=0.1, n_jobs=8)
        model3 = GaussianNB()
        model8 = RandomForestRegressor(n_estimators=200, max_features=5, random_state=1)

        model4 = KNeighborsClassifier(n_neighbors=300)
        model5 = SVC(kernel='rbf')
        model6 = SVC(kernel='linear', C=1, gamma='scale')
        model7 = SVC(kernel='poly')
        best_model = None
        best_model_score = 0
        models = [model1, model2, model3, model4, model5, model6, model7, model8]
        # models = [model3, model8]
        # mx = X[:5]
        # print(mx)
        # origy = Y[:5]
        # my = origy
        label_encoder = LabelEncoder()
        for i, model in enumerate(models):
            start_time = time.time()
            i += 1
            print("Training {} started....".format(model))
            model_report.append("Training {} started....".format(model))
            if model in [model1, model2, model3, model4, model8]:
                Y1 = label_encoder.fit_transform(Y)
                X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.000001, random_state=1)
                X1_train, Z_test, y1_train, z_test = train_test_split(X, Y1, test_size=0.99, random_state=1)
                # my = Y1[:5]
            else:
                X_train = orig_X_train
                y_train = orig_y_train
                Z_test = orig_Z_test
                z_test = orig_z_test
                # my = origy
            
            if model == model2:
                model.fit(X_train, y_train, 
                early_stopping_rounds=25, 
                eval_set=[(Z_test, z_test)],
                verbose=False)
            else:
                model.fit(X_train, y_train)
            
            print("Training completed !!!!")
            score = model.score(Z_test, z_test)*100
            if score > best_model_score:
                best_model_score = score
                best_model = model
            print("Model test accuracy: {:.3f} %".format(score))
            end_time = time.time()
            print("Total time taken for training: {} s".format(end_time-start_time))
            model_report.append(f'Model test accuracy: {model.score(Z_test, z_test)*100:.3f}%')
            model_report.append("Total time taken for training: {} s".format(end_time-start_time))
            model_report.append("=========\n\n")
        
        
        self.save_model_report(model_report)
        self.save_object(best_model, naive_bayes_model)
        self.save_object(model8, random_forest_model)
        self.save_object(label_encoder, model_encoder)

        


if __name__ == '__main__':
    # GenerateModel().vectorization_model()

    GenerateModel().get_best_model()