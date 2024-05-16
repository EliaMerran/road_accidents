import os

import numpy as np
import pandas as pd
import json


def combine_data(directory, search_name, save_path=None):
    dataframes = []
    # Recursive function to search for files containing the specified file_name
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            if search_name in file.lower():
                file_path = os.path.join(root, file)
                df = pd.read_csv(
                    file_path, index_col='pk_teuna_fikt')
                dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes)
        combined_df.drop_duplicates(inplace=True)
        if save_path is not None:
            combined_df.to_csv(save_path)
        return combined_df
    else:
        print(f"No files containing '{search_name}' found.")


def get_accidents_data(save_path=None):
    with_injuries = combine_data('data/raw data/accidents with injuries 2005-2021/', 'accdata.csv')
    without_injuries = combine_data('data/raw data/accidents without injuries 2005-2021/',
                                    'accdata.csv')
    without_injuries['HUMRAT_TEUNA'] = 4
    combined_df = pd.concat([with_injuries, without_injuries])
    # combined_df = combined_df.drop_duplicates(subset="pk_teuna_fikt").set_index("pk_teuna_fikt")
    combined_df = combined_df.drop_duplicates()

    if save_path is not None:
        combined_df.to_csv(save_path)
    return combined_df


def get_city_info(config, sort_by_population=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    city_codes = pd.read_excel(os.path.join(script_dir,config["CITY_CODES_PATH"]), engine='openpyxl')
    if sort_by_population:
        city_codes.sort_values(by=['סך הכל אוכלוסייה 2021'], ascending=False, inplace=True)
    return city_codes


def get_city_mapping(config, save_path=None):

    num_cities = config["NUM_CITIES"]
    city_info = get_city_info(config, sort_by_population=True)
    city_mapping = city_info[:num_cities].set_index('סמל יישוב')['שם יישוב באנגלית'].to_dict()
    if save_path is not None:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as json_file:
            json.dump(city_mapping, json_file)
    return city_mapping


def get_cities_data(config, data=None, save_path=None):
    if data is None:
        data = get_accidents_data()
    data = data[data.SHNAT_TEUNA.between(config["START_YEAR"], config["END_YEAR"])]
    city_info = get_city_info(config, sort_by_population=True)
    city_mapping = city_info[:config["NUM_CITIES"]].set_index('סמל יישוב')['שם יישוב באנגלית'].to_dict()
    data_cities = data[data['SEMEL_YISHUV'].isin(city_mapping.keys())]
    data_cities = data_cities[data_cities.STATUS_IGUN == 1]
    data_cities['BAKARA'] = data_cities['BAKARA'].fillna(0)
    if save_path is not None:
        data_cities.to_csv(save_path)
    return data_cities


def split_X_y(df):
    X = df.drop(columns=['dangerous', 'test_year'])
    y = df['dangerous'].astype(int)
    return X, y


def prepare_clusters_data_for_xgboost(clusters_data, config):
    clusters_data.drop(columns=config["DROP_COLUMNS"], inplace=True)
    cats = clusters_data.select_dtypes(exclude=np.number).columns.tolist()

    # Convert to Pandas category
    for col in cats:
        # print(col, clusters_data[col].dtype)
        clusters_data[col] = clusters_data[col].astype('category')


def split_train_test(clusters_data, config):
    df = clusters_data.copy()
    prepare_clusters_data_for_xgboost(df, config)
    test_start = config["XGBOOST_TEST_START"]
    test_end = config["XGBOOST_TEST_END"]
    train_start = config["START_YEAR"]
    train_end = test_start - 1
    X_train, y_train = split_X_y(df[df.test_year.between(train_start, train_end)])
    X_test, y_test = split_X_y(df[df.test_year.between(test_start, test_end)])
    return X_train, y_train, X_test, y_test


def create_attribute_dict_israel(path_codebook, save_path=None):
    df = pd.read_csv(path_codebook)
    attributes_dict = {}
    for var_value in df['var'].unique():
        var_df = df[df['var'] == var_value]
        var_dict = dict(zip(var_df['code'], var_df['value']))
        attributes_dict[var_value] = var_dict

    # Add 4: 'property damage' to HUMRAT_TEUNA attribute
    attributes_dict['HUMRAT_TEUNA'][4] = 'property damage'
    # Change 'IGUN_MEKUBAZ' key to 'STATUS_IGUN'
    attributes_dict['STATUS_IGUN'] = attributes_dict.pop('IGUN_MEKUBAZ')
    for attribute in attributes_dict.keys():
        if 0 not in attributes_dict[attribute] and 'unknown' not in attributes_dict[attribute].values():
            attributes_dict[attribute][0] = 'unknown'
    if save_path is not None:
        with open(save_path, 'w') as json_file:
            json.dump(attributes_dict, json_file)
    return attributes_dict


def create_attribute_dict_uk(path_codebook=r'C:\code\road_accidents\data\raw data\uk\
dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2023.xlsx', save_path=None):
    df = pd.read_excel(path_codebook)
    df = df[(df['table'] == 'Accident') & (df['code/format'].notnull())]
    attribute_dict = {}

    # Iterate over unique 'var' values
    for var_value in df['field name'].unique():
        var_df = df[df['field name'] == var_value]
        var_dict = dict(zip(var_df['code/format'], var_df['label']))
        attribute_dict[var_value] = var_dict
    del_keys = ['legacy_collision_severity', 'did_police_officer_attend_scene_of_collision']
    for key in del_keys:
        attribute_dict.pop(key)
    # attribute_dict['HUMRAT_TEUNA'] = attribute_dict['accident_severity']
    if save_path is not None:
        with open(save_path, 'w') as json_file:
            json.dump(attribute_dict, json_file)
    return attribute_dict


def create_attribute_dict(country, path_codebook, save_path=None):
    if country == "UK":
        return create_attribute_dict_uk(path_codebook=path_codebook, save_path=save_path)
    elif country == "ISRAEL":
        return create_attribute_dict_israel(path_codebook=path_codebook, save_path=save_path)


def load_config(use_uk=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load base configuration
    with open(os.path.join(script_dir, "configurations/base_config.json")) as f:
        base_config = json.load(f)
    # Load configuration from a file
    with open(os.path.join(script_dir, "configurations/uk_config.json") if use_uk else
              os.path.join(script_dir, "configurations/israel_config.json")) as f:
        country_config = json.load(f)
    # Merge base and country-specific configurations
    config = {**base_config, **country_config}
    # Calculate END_TRAIN_YEAR
    config["END_TRAIN_YEAR"] = config["END_YEAR"] - config["DBSCAN_TRAIN_INTERVAL"] - config[
        "DBSCAN_TEST_INTERVAL"] + 2
    config["ATTRIBUTE_DICT"] = create_attribute_dict(config["COUNTRY"],os.path.join(script_dir, config["CODEBOOK_PATH"]))
    return config


def load_data(config):
    data = pd.read_csv(config["DATA_PATH"], index_col=config["INDEX_FEATURE"])
    if config["COUNTRY"] == "ISRAEL":
        data = data[data.STATUS_IGUN == 1]
    data.dropna(subset=[config["X_FEATURE"], config["Y_FEATURE"]], inplace=True)
    data = data[data[config["YEAR_FEATURE"]].between(config["START_YEAR"], config["END_YEAR"])]
    return data


def load_cities_data(config):
    cities_data = pd.read_csv(config["CITY_DATA_PATH"], index_col=config["INDEX_FEATURE"])
    if config["COUNTRY"] == "ISRAEL":
        cities_data = cities_data[cities_data.STATUS_IGUN == 1]
    cities_data.dropna(subset=[config["X_FEATURE"], config["Y_FEATURE"]], inplace=True)
    cities_data = cities_data[cities_data[config["YEAR_FEATURE"]].between(config["START_YEAR"], config["END_YEAR"])]
    return cities_data


def uk_cities_data(config, save_path=None):
    data = load_data(config)
    # print data columns dtype
    print(data.dtypes)
    # Load mapping
    with open("data/United Kingdom/city_mapping.json") as f:
        city_mapping = json.load(f)
    # turn keys to int
    city_mapping = {int(k): v for k, v in city_mapping.items()}
    cities_data = data[data['local_authority_district'].isin(city_mapping.keys())]
    print(cities_data.head())
    if save_path is not None:
        cities_data.to_csv(save_path)
    return cities_data


if __name__ == '__main__':
    # Israel
    config_israel = load_config(use_uk=False)
    # save israel city mapping
    get_city_mapping(config_israel, 'data/Israel/city_mapping.json')

    # save data
    # uk_cities_data(config,'data/United Kingdom/Accidents_cities.csv')
