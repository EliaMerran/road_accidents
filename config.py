import numpy as np
import pandas as pd

EPS = 30
MIN_SAMPLES = 5
MIN_CLUSTER_SIZE = 5
START_YEAR = 2008
END_YEAR = 2019
TRAIN_INTERVAL = 3
END_TRAIN_YEAR = END_YEAR - TRAIN_INTERVAL + 1
NUM_CITIES = 20
ACCIDENT_CODEBOOK_PATH = 'data/raw data/accident_codebook.csv'

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

# Add 0:'unknown' to all attributes
for attribute in ATTRIBUTES_DICT.keys():
    if 0 not in ATTRIBUTES_DICT[attribute]:
        ATTRIBUTES_DICT[attribute][0] = 'missing data'

# Change 'IGUN_MEKUBAZ' key to 'STATUS_IGUN'
ATTRIBUTES_DICT['STATUS_IGUN'] = ATTRIBUTES_DICT.pop('IGUN_MEKUBAZ')