from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

import config
import preprocess
import utilities
import xgboost as xgb
import numpy as np


def split_X_y(df):
    X = df.drop(columns=['type', 'train_start'])
    y = df['type']
    return X, y


if __name__ == '__main__':
    # Load data
    data = utilities.get_cities_data()
    # Preprocess data
    data = preprocess.preprocess(data, start_year=config.START_YEAR, end_year=config.END_YEAR,
                                 train_interval=config.TRAIN_INTERVAL)
    data = data.drop(columns=['cluster', 'geometry', 'test_severe'])
    # Extract text features
    cats = data.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        data[col] = data[col].astype('category')

    # Split data
    X_test, y_test = split_X_y(data[data.train_start.between(2015, 2016)])
    X_val, y_val = split_X_y(data[data.train_start.between(2014, 2014)])
    X_train, y_train = split_X_y(data[data.train_start.between(2008, 2013)])

    # y_test_encoded = OrdinalEncoder().fit_transform(y_test)
    # y_val_encoded = OrdinalEncoder().fit_transform(y_val)
    # y_train_encoded = OrdinalEncoder().fit_transform(y_train)
    #
    #
    # # Fit model
    # # Create regression matrices
    # dtrain_reg = xgb.DMatrix(X_train, y_train_encoded, enable_categorical=True)
    # dtest_reg = xgb.DMatrix(X_val, y_val_encoded, enable_categorical=True)
    #
    # # Define hyperparameters
    # params = {"objective": "multi:softprob", "tree_method": "gpu_hist"}
    #
    # n = 100
    # model = xgb.train(
    #     params=params,
    #     dtrain=dtrain_reg,
    #     num_boost_round=n,
    # )
    # # Evaluate model
    # preds = model.predict(dtest_reg)
    # rmse = mean_squared_error(y_test, preds, squared=False)
    #
    # print(f"RMSE of the base model: {rmse:.3f}")

    # Plot results

    # Save results
