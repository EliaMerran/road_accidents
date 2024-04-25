import numpy as np
import pandas as pd
import json
import os

EPS = 30
MIN_SAMPLES = 3
MIN_CLUSTER_SIZE = 1  # 3
NUM_ESTIMATORS = 10
MAX_DEPTH = 2
THRESHOLD = 0.1
START_YEAR = 2008
END_YEAR = 2019
TRAIN_INTERVAL = 5
TEST_INTERVAL = 2
END_TRAIN_YEAR = END_YEAR - TRAIN_INTERVAL - TEST_INTERVAL + 2
NUM_CITIES = 20
RANDOM_STATE = 42


# Get the absolute path of the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path from the script's directory
ACCIDENT_CODEBOOK_PATH = os.path.join(script_dir, 'data/raw data/accident_codebook.csv')
CITY_MAPPING_PATH = os.path.join(script_dir,'data/processed data/city_mapping.json')
CITY_DATA_PATH = os.path.join(script_dir,'data/processed data/cities_accidents.csv')
CITY_CODES_PATH = os.path.join(script_dir,'data/raw data/city_codes.xlsx')
DROP_COLUMNS = ['cluster', 'geometry_x','geometry_y', 'type', 'test_severe', 'train_index', 'test_index']

df = pd.read_csv(ACCIDENT_CODEBOOK_PATH)

ATTRIBUTES_DICT = {}

# Iterate over unique 'var' values
for var_value in df['var'].unique():
    # Filter DataFrame for the current 'var' value
    var_df = df[df['var'] == var_value]

    # Create a dictionary for the current 'var' value
    var_dict = dict(zip(var_df['code'], var_df['value']))

    # Update the main dictionary with the 'var' dictionary
    ATTRIBUTES_DICT[var_value] = var_dict

# Add 4: 'property damage' to HUMRAT_TEUNA attribute
ATTRIBUTES_DICT['HUMRAT_TEUNA'][4] = 'property damage'

# Load dictionary from the JSON file
with open(CITY_MAPPING_PATH, 'r') as json_file:
    city_mapping = json.load(json_file)
    city_mapping = {int(key): value for key, value in city_mapping.items()}
ATTRIBUTES_DICT['SEMEL_YISHUV'] = city_mapping


# Change 'IGUN_MEKUBAZ' key to 'STATUS_IGUN'
ATTRIBUTES_DICT['STATUS_IGUN'] = ATTRIBUTES_DICT.pop('IGUN_MEKUBAZ')


# Add 0:'unknown' to all attributes
for attribute in ATTRIBUTES_DICT.keys():
    if 0 not in ATTRIBUTES_DICT[attribute] and 'unknown' not in ATTRIBUTES_DICT[attribute].values():
        ATTRIBUTES_DICT[attribute][0] = 'unknown'
