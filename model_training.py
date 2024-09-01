import json
import utilities
import preprocess
import numpy as np
import xgboost as xgb
import pandas as pd


def train_model(data, config, save_path=None):
    clusters_data = preprocess.preprocess(data, config)
    clusters_data.to_csv(save_path.replace(".json", "_data.csv"), index=False)
    # Split data
    X_train, y_train, X_test, y_test = utilities.split_train_test(clusters_data, config)
    # Fit model
    xgb_classifier = xgb.XGBClassifier(n_estimators=config["NUM_ESTIMATORS"], objective='binary:logistic',
                                       tree_method='hist', max_depth=config["MAX_DEPTH"], enable_categorical=True,
                                       random_state=config["RANDOM_STATE"])
    xgb_classifier.fit(X_train, y_train)

    if save_path:
        xgb_classifier.save_model(save_path)
        file_name = save_path.replace(".json", "_configuration.json")
        with open(file_name, "w") as f:
            json.dump(config, f)
    return xgb_classifier


if __name__ == '__main__':
    # config_israel = utilities.load_config()
    # config_uk = utilities.load_config(use_uk=True)
    # configs = [config_israel, config_uk]
    # for conf in configs:
    #     data = utilities.load_data(conf)
    #     cities_data = utilities.load_cities_data(conf)
    #     train_model(data, conf, save_path=conf["MODEL_PATH"])
    #     train_model(cities_data, conf, save_path=conf["CITY_MODEL_PATH"])

    config_israel = utilities.load_config()
    cities_data = utilities.load_cities_data(config_israel)
    train_model(cities_data, config_israel, save_path='models/israel_cities_model_3_1_severe_year_split.json')