import utilities
import preprocess
import config

import numpy as np
import xgboost as xgb
import pandas as pd
from dbscan_hyperparameter_tuning import calculate_param_from_accidents_density


def prediction_model(data, city_mapping, city_attributes, test_start, test_end, min_samp_formula=None,
                     min_samp_args=None, eps_formula=None, eps_args=None):
    list_of_df = []
    for city_code, city_name in city_mapping.items():
        city_data = data[data['SEMEL_YISHUV'] == city_code]
        min_samp = config.MIN_SAMPLES
        eps = config.EPS
        if min_samp_formula:
            min_samp = round(calculate_param_from_accidents_density(data, city_attributes, city_name, city_code,
                                                              min_samp_formula, *min_samp_args))
            if min_samp < 2:
                min_samp = 2
        if eps_formula:
            eps = calculate_param_from_accidents_density(data, city_attributes, city_name, city_code,
                                                         eps_formula, *eps_args)
            if eps < 1:
                eps = 1
        clusters_data = preprocess.preprocess(city_data, start_year=config.START_YEAR, end_year=config.END_YEAR,
                                              train_interval=config.TRAIN_INTERVAL, test_interval=config.TEST_INTERVAL,
                                              min_samp=min_samp, min_cluster_size=1, cluster_eps=eps, test_eps=eps)
        processes_data = clusters_data.drop(columns=config.DROP_COLUMNS)
        list_of_df.append(processes_data)
    processes_data = pd.concat(list_of_df)
    # Extract text features
    cats = processes_data.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        processes_data[col] = processes_data[col].astype('category')

    # Split data
    X_train, y_train, X_test, y_test = utilities.split_train_test(processes_data, test_start, test_end)

    # Fit model
    xgb_classifier = xgb.XGBClassifier(n_estimators=config.NUM_ESTIMATORS, objective='binary:logistic',
                                       tree_method='hist', max_depth=config.MAX_DEPTH, enable_categorical=True,
                                       random_state=config.RANDOM_STATE)
    xgb_classifier.fit(X_train, y_train)
    return xgb_classifier, X_test, y_test

if __name__ == '__main__':
    pass
