import utilities
import preprocess


def setup_israel_data():
    """
    This function sets up the data for the israeli dataset
    With injuries accidents data should be in data/raw data/accidents with injuries 2005-2021
    Without injuries accidents data should be in data/raw data/accidents without injuries 2005-2021
    city_codes.xlsx should be in data/raw data
    :return:
    """
    config = utilities.load_config(use_uk=False)
    utilities.get_accidents_data(save_path=config["DATA_PATH"])
    data = utilities.load_data(config)
    clusters_data = preprocess.preprocess(data,config,save_path=config["CLUSTERS_DATA_PATH"])
    cities_data = utilities.get_cities_data(config, data, save_path=config["CITY_DATA_PATH"])
    clusters_cities_data = preprocess.preprocess(cities_data,config,save_path=config["CITY_CLUSTERS_DATA_PATH"])


def setup_uk_data():
    """
    dft-road-casualty-statistics-collision-1979-latest-published-year.csv file should be in
    data/United Kingdom as Accidents.csv
    :return:
    """
    config = utilities.load_config(use_uk=True)
    data = utilities.load_data(config)
    clusters_data = preprocess.preprocess(data,config,save_path=config["CLUSTERS_DATA_PATH"])
    cities_data = utilities.uk_cities_data(config, save_path=config["CITY_DATA_PATH"])
    clusters_cities_data = preprocess.preprocess(cities_data,config,save_path=config["CITY_CLUSTERS_DATA_PATH"])


if __name__ == '__main__':
    setup_israel_data()
    setup_uk_data()