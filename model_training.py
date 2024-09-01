import json
import utilities
import preprocess
import xgboost as xgb


def train_model(data, config, save_path=None):
    """
        Trains an XGBoost classifier on clustered accident data using specified configuration settings.

        The function preprocesses the input data, splits it into training and test sets,
        and trains an XGBoost classifier with parameters defined in the configuration.
        Optionally, it saves the trained model, processed data, and configuration to specified paths.

        Args:
            data (DataFrame): Input data containing features for model training.
            config (dict): Configuration dictionary containing settings for preprocessing, model parameters, and training.
            save_path (str, optional): Path to save the trained model, processed data, and configuration. Defaults to None.

        Returns:
            XGBClassifier: The trained XGBoost classifier model.
        """
    clusters_data = preprocess.preprocess(data, config)
    # Split data
    X_train, y_train, X_test, y_test = utilities.split_train_test(clusters_data, config)
    # Fit model
    xgb_classifier = xgb.XGBClassifier(n_estimators=config["NUM_ESTIMATORS"], objective='binary:logistic',
                                       tree_method='hist', max_depth=config["MAX_DEPTH"], enable_categorical=True,
                                       random_state=config["RANDOM_STATE"])
    xgb_classifier.fit(X_train, y_train)

    if save_path:
        xgb_classifier.save_model(save_path)
        data_path = save_path.replace(".json", "_data.csv")
        clusters_data.to_csv(data_path, index=False)
        file_name = save_path.replace(".json", "_configuration.json")
        with open(file_name, "w") as f:
            json.dump(config, f)
    return xgb_classifier


if __name__ == '__main__':
    config_israel = utilities.load_config()
    config_uk = utilities.load_config(use_uk=True)
    configs = [config_israel, config_uk]
    for conf in configs:
        data = utilities.load_data(conf)
        cities_data = utilities.load_cities_data(conf)
        train_model(data, conf, save_path=conf["MODEL_PATH"])
        train_model(cities_data, conf, save_path=conf["CITY_MODEL_PATH"])