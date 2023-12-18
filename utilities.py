import os
import pandas as pd

import config


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
    without_injuries = combine_data('data/raw data/accidents without injuries 2005-2021/', 'accdata.csv')
    without_injuries['HUMRAT_TEUNA'] = 4
    combined_df = pd.concat([with_injuries, without_injuries])

    # combined_df = combined_df.drop_duplicates(subset="pk_teuna_fikt").set_index("pk_teuna_fikt")
    combined_df = combined_df.drop_duplicates()

    if save_path is not None:
        combined_df.to_csv(save_path)
    return combined_df


def get_city_info(sort_by_population=True):
    city_codes = pd.read_excel(r'data/raw data/city_codes.xlsx', engine='openpyxl')
    if sort_by_population:
        city_codes.sort_values(by=['סך הכל אוכלוסייה 2021'], ascending=False, inplace=True)
    return city_codes


def get_city_mapping(num_cities=config.NUM_CITIES):
    city_info = get_city_info(sort_by_population=True)
    city_mapping = city_info[:num_cities].set_index('סמל יישוב')['שם יישוב באנגלית'].to_dict()
    return city_mapping


def get_cities_data(save_path=None):
    data = get_accidents_data()
    data = data[data.SHNAT_TEUNA.between(config.START_YEAR, config.END_YEAR)]
    city_info = get_city_info(sort_by_population=True)
    city_mapping = city_info[:20].set_index('סמל יישוב')['שם יישוב באנגלית'].to_dict()
    data_cities = data[data['SEMEL_YISHUV'].isin(city_mapping.keys())]
    data_cities = data_cities[data_cities.STATUS_IGUN == 1]
    if save_path is not None:
        data_cities.to_csv(save_path)
    return data_cities


if __name__ == '__main__':
    # save data
    get_cities_data(r'data\processed data\cities_accidents.csv')
    # get_accidents_data(r'data\processed data\all_accidents.csv')