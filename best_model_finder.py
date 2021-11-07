import time
import pandas as pd
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils import pickle_object, load_nlp
from config import (token_data_file, sent_data_file, feature_list, model_report_file,
                    categorical_nb_model, svm_model, linear_svm_model, svr_model, linear_svr_model,
                    gaussian_nb_model, random_forest_model, regressor_model, knn_model, bernouli_nb_model)


class GenerateModel():
    def __init__(self):
        self.nlp = load_nlp()
        self.vec_file = pd.read_csv(sent_data_file)
        self.rf_file = pd.read_csv(token_data_file)

    def save_object(self, object, filename):
        pickle_object(object, filename)

    def save_model_report(self, info):
        with open(model_report_file, "w+") as fp:
            fp.write("\n".join(info))        

    def test_for_best_models(self):
        label_encoder = LabelEncoder()
        Y = self.rf_file.disease
        model_report = []
        features = self.rf_file.columns.drop('disease')
        self.save_object(features, feature_list)
        X = self.rf_file[features]
        default_X_train, _, default_y_train, _ = train_test_split(X, Y, test_size=0.000001, random_state=1)
        _, default_X_test, _, default_y_test = train_test_split(X, Y, test_size=0.99, random_state=1)
        # encoding string into digits
        Y1 = label_encoder.fit_transform(Y)
        encoded_X_train, _, encoded_y_train, _ = train_test_split(X, Y1, test_size=0.000001, random_state=1)
        _, encoded_X_test, _, encoded_y_test = train_test_split(X, Y1, test_size=0.99, random_state=1)
        
        scaler = StandardScaler()
        scaler.fit(default_X_train)
        default_X_train = scaler.transform(default_X_train)

        # Models that need label encoding
        model1 = RandomForestRegressor(n_estimators=200, max_features=5, random_state=1)
        model2 = XGBRegressor(n_estimators=250, learning_rate=0.1, n_jobs=8)
        model4 = KNeighborsClassifier(n_neighbors=300)
        # SVM models
        model5 = SVC()
        model6 = LinearSVC()
        model7 = SVR()
        model11 = LinearSVR()
        
        # Models that without label encoding
        model3 = GaussianNB()
        model9 = BernoulliNB()
        model10 = CategoricalNB()

        best_model = None
        best_model_score = 0
        models = [model1, model2, model3, model4, model5, model6, model7, model9, model10, model11]
        for i, model in enumerate(models):
            start_time = time.time()
            i += 1
            print("Training {} started....".format(model))
            model_report.append("Training {} started....".format(model))

            if model in [model1, model2, model4, model5, model6, model7, model11]:
                X_train = encoded_X_train
                y_train = encoded_y_train
                X_test = encoded_X_test
                y_test = encoded_y_test
            else:
                X_train = default_X_train
                y_train = default_y_train
                X_test = default_X_test
                y_test = default_y_test
            

            if model in [model3, model9, model10]:
                model.fit(X,Y)
            elif model == model2:
                model.fit(X_train, y_train, 
                early_stopping_rounds=25, 
                eval_set=[(X_test, y_test)],
                verbose=False)
            else:
                model.fit(X_train, y_train)

            print("Training completed !!!!")
            score = model.score(X_test, y_test)*100
            if score >= best_model_score:
                best_model_score = score
                best_model = model
            print("Model test accuracy: {:.3f} %".format(score))
            end_time = time.time()
            print("Total time taken for training: {} s".format(end_time-start_time))
            model_report.append(f'Model test accuracy: {score:.3f}%')
            model_report.append("Total time taken for training: {} s".format(end_time-start_time))
            model_report.append("=========\n\n")
        
        
        self.save_model_report(model_report)
        self.save_object(model1, random_forest_model)
        self.save_object(model2, regressor_model)
        self.save_object(model3, gaussian_nb_model)
        self.save_object(model4, knn_model)
        self.save_object(model5, svm_model)
        self.save_object(model6, linear_svm_model)
        self.save_object(model7, svr_model)
        self.save_object(model11, linear_svr_model)
        self.save_object(model9, bernouli_nb_model)
        self.save_object(model10, categorical_nb_model)
        self.save_object(label_encoder, model_encoder)

        


if __name__ == '__main__':
    GenerateModel().test_for_best_models()