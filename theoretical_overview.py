import pandas as pd
import numpy as np
import geopandas as gpd

import utilities
import preprocess


# creates a table of the following format:
# city, population, total accidents, (of them) property damage, light, severe, casualty
def accident_statistics_by_city_table(data, config, city_mapping, output_path=None):
    # Create city mapping
    city_info = utilities.get_city_info(config=config, sort_by_population=True).head(config["NUM_CITIES"])
    # Group and pivot data
    pivot = (
        data.groupby([config["CITY_FEATURE"], config["YEAR_FEATURE"], config["SEVERE_FEATURE"]])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Mean over years
    pivot = pivot.groupby(config["CITY_FEATURE"]).mean().reset_index()
    # Map city names and calculate total accidents
    pivot['city'] = pivot[config["CITY_FEATURE"]].map(city_mapping)
    pivot['total_accidents'] = pivot[[1, 2, 3, 4]].sum(axis=1)
    pivot['population'] = pivot[config["CITY_FEATURE"]].map(city_info.set_index('סמל יישוב')['סך הכל אוכלוסייה 2021'])

    # Rearrange columns
    column_order = ['city', 'population', 'total_accidents', 4, 3, 2, 1]
    pivot = pivot[column_order]

    # Rename columns
    pivot.columns = ['city', 'population', 'total_accidents', 'property_damage', 'light', 'severe', 'casualty']
    pivot = pivot.round(2)
    # Save to CSV if requested
    if output_path:
        pivot.to_csv(output_path, index=False)
    return pivot


# creates a table of the following format:
# city, mean_accidents, mean_minor_accidents, mean_severe_accidents, num_clusters,
# Consistent Severity, Improved Safety, Severe Turnaround, Stable Safety, severe_turnaround_probability	,
# consistent_severe_probability, cluster size, average/std/min/max number of accidents per cluster (count)
def accident_clusters_statistics_by_city_table(data, config, output_path=None):
    min_cluster_size = config["MIN_CLUSTER_SIZE"]
    city_mapping = utilities.get_city_mapping(config=config)
    # not the prettiest with ramzor
    attribute = 'RAMZOR'
    list_of_dfs = []
    for city_code, city_name in city_mapping.items():
        city_data = data[data[config["CITY_FEATURE"]] == city_code].copy()
        mean_accidents = city_data.groupby(config["YEAR_FEATURE"]).size().mean()
        mean_minor_accidents = city_data[city_data[config["SEVERE_FEATURE"]] >= 3].groupby(
            config["YEAR_FEATURE"]).size().mean()
        mean_severe_accidents = city_data[city_data[config["SEVERE_FEATURE"]] < 3].groupby(
            config["YEAR_FEATURE"]).size().mean()

        num_clusters = []
        cluster_size = []
        min_cluster_count = []
        average_cluster_count = []
        max_cluster_count = []
        std_cluster_count = []
        df_final = pd.DataFrame()
        for train_start in range(config["START_YEAR"], config["END_TRAIN_YEAR"]):
            train_end = train_start + 2
            test_start = train_end + 1
            test_end = test_start
            city_train, city_test = preprocess.cluster_frame(frame=city_data, config=config, train_start=train_start,
                                                             train_end=train_end,
                                                             test_start=test_start, test_end=test_end)
            city_train = city_train[city_train.cluster != -1]
            clusters = city_train.groupby('cluster').size().reset_index(name='count')
            clusters = clusters[clusters['count'] >= min_cluster_size]
            # clusters = clusters[clusters['cluster'] != -1]
            num_clusters.append(len(clusters))
            describe = clusters['count'].describe()
            min_cluster_count.append(describe['min'])
            average_cluster_count.append(describe['mean'])
            max_cluster_count.append(describe['max'])
            std_cluster_count.append(describe['std'])

            train = gpd.GeoDataFrame(city_train[[config["X_FEATURE"], config["Y_FEATURE"], "cluster"]],
                                     geometry=gpd.points_from_xy(city_train.X, city_train.Y, crs=config["CRS"]))
            cluster_size.append(train.dissolve(by='cluster').minimum_bounding_radius().mean())
            res = preprocess.get_clusters_by_attribute(train_frame=city_train, test_frame=city_test,
                                                       attribute=attribute,
                                                       attribute_dict=config["ATTRIBUTE_DICT"][attribute],
                                                       config=config)
            df_grouped = res.groupby(['type']).size().reset_index(name='count')
            df_final = pd.concat([df_final, df_grouped.set_index('type').transpose()])
            df_final.fillna(0, inplace=True)

            # if df_final columns are not ['Consistent Severity', 'Improved Safety', 'Severe Turnaround',
            #        'Stable Safety']: add the missing columns with 0
            desired_columns = ['Consistent Severity', 'Improved Safety', 'Severe Turnaround', 'Stable Safety']
            missing_columns = set(desired_columns) - set(df_final.columns)
            for column in missing_columns:
                df_final[column] = 0

        df_final.columns.name = None
        df_final.reset_index(drop=True, inplace=True)
        df_final = df_final.mean(axis=0).to_frame().transpose()

        cluster_size = np.array(cluster_size)
        cluster_size = np.nan_to_num(cluster_size, nan=0)

        min_cluster_count = np.array(min_cluster_count)
        min_cluster_count = np.nan_to_num(min_cluster_count, nan=0)

        average_cluster_count = np.array(average_cluster_count)
        average_cluster_count = np.nan_to_num(average_cluster_count, nan=0)

        max_cluster_count = np.array(max_cluster_count)
        max_cluster_count = np.nan_to_num(max_cluster_count, nan=0)

        std_cluster_count = np.array(std_cluster_count)
        std_cluster_count = np.nan_to_num(std_cluster_count, nan=0)

        df_final.insert(0, 'city', city_name)
        df_final.insert(1, 'mean_accidents', mean_accidents)
        df_final.insert(2, 'mean_minor_accidents', mean_minor_accidents)
        df_final.insert(3, 'mean_severe_accidents', mean_severe_accidents)
        df_final.insert(4, 'num_clusters', np.mean(num_clusters))
        df_final.insert(5, 'mean_cluster_size', np.mean(cluster_size))
        df_final.insert(6, 'min_cluster_count', np.mean(min_cluster_count))
        df_final.insert(7, 'average_cluster_count', np.mean(average_cluster_count))
        df_final.insert(8, 'max_cluster_count', np.mean(max_cluster_count))
        df_final.insert(9, 'std_cluster_count', np.mean(std_cluster_count))
        df_final['severe_turnaround_probability'] = (df_final['Severe Turnaround'] / (
                df_final['Severe Turnaround'] + df_final['Stable Safety'])) * 100
        df_final['consistent_severe_probability'] = (df_final['Consistent Severity'] / (
                df_final['Consistent Severity'] + df_final['Improved Safety'])) * 100

        df_final = df_final.round(2)
        list_of_dfs.append(df_final)

    result_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path:
        result_df.to_csv(output_path, index=False)
    return result_df


# # creates a table of the following format:
# (Exactly what you sent last week in attributes_clusters_summary_intervals.csv)
def accident_clusters_statistics_by_attribute_table(data, config, output_path=None):
    min_cluster_size = config["MIN_CLUSTER_SIZE"]
    list_of_dfs = []
    num_clusters_lst = []
    for train_start in range(config["START_YEAR"], config["END_TRAIN_YEAR"]):
        train_end = train_start + 2
        test_start = train_end + 1
        test_end = test_start
        city_train, city_test = preprocess.cluster_frame(frame=data, config=config, train_start=train_start, train_end=train_end,
                                                         test_start=test_start, test_end=test_end)
        clusters = city_train.groupby('cluster').size().reset_index(name='size')
        clusters = clusters[clusters['size'] >= min_cluster_size]
        n_cluster = len(clusters[clusters['cluster'] != -1])
        num_clusters_lst.append(n_cluster)
        for attribute in config["ATTRIBUTE_DICT"].keys():
            res = preprocess.get_clusters_by_attribute(train_frame=city_train, test_frame=city_test,
                                                       attribute=attribute,
                                                       attribute_dict=config["ATTRIBUTE_DICT"][attribute],
                                                       config=config)
            # Accident Percentage

            df_per = res.groupby([f'{attribute}_majority_vote']).sum()
            df_per.reset_index(inplace=True)
            df_per.drop(columns=['cluster', 'cluster_size', 'train_severe', 'test_severe', 'type'], inplace=True)
            options = df_per[f'{attribute}_majority_vote'].unique()
            # Iterate over each option and create a new column for each
            df_per['Accidents Percentage'] = np.nan
            for option in options:
                conditions = (df_per[f'{attribute}_majority_vote'] == option)
                df_per['Accidents Percentage'] = np.where(conditions,
                                                          (df_per[option] / df_per[options].sum(axis=1)) * 100,
                                                          df_per['Accidents Percentage'])

            df_grouped = res.groupby([f'{attribute}_majority_vote', 'type']).size().reset_index(name='count')
            # Calculate the total count for each type
            type_totals = df_grouped.groupby('type')['count'].transform('sum')
            # Calculate the percentage column
            df_grouped['percentage'] = ((df_grouped['count'] / type_totals) * 100)
            df_grouped.rename(columns={'type': 'cluster_type', f'{attribute}_majority_vote': 'attribute_majority_vote'},
                              inplace=True)
            df_grouped['attribute_majority_vote'] = df_grouped['attribute_majority_vote'].apply(
                lambda x: f"{attribute} - {x}")
            df_grouped['min_cluster_size'] = min_cluster_size
            df_grouped = df_grouped.pivot(index='attribute_majority_vote', columns='cluster_type',
                                          values='count').fillna(0).astype(int)
            cluster_count = df_grouped.sum(axis=1)
            df_grouped.insert(0, 'clusters_count', cluster_count)
            df_grouped.insert(1, 'clusters_percent', cluster_count / n_cluster * 100)
            df_grouped.reset_index(inplace=True)
            df_grouped.columns.name = None
            df_grouped['severe_turnaround_probability'] = (df_grouped['Severe Turnaround'] / (
                    df_grouped['Severe Turnaround'] + df_grouped['Stable Safety'])) * 100
            df_grouped['consistent_severe_probability'] = (df_grouped['Consistent Severity'] / (
                    df_grouped['Consistent Severity'] + df_grouped['Improved Safety'])) * 100
            df_grouped['Accidents Percentage'] = df_per['Accidents Percentage']
            df_grouped = df_grouped.round(2)
            list_of_dfs.append(df_grouped)

    # Concatenate the DataFrames along the rows
    merged_df = pd.concat(list_of_dfs, ignore_index=True)
    # Group by the 'attribute_majority_vote' column and calculate the mean for each group
    result_df = merged_df.groupby('attribute_majority_vote').mean().reset_index()
    split = result_df['attribute_majority_vote'].str.split(' - ', expand=True, n=1)
    result_df.insert(0, 'attribute', split[0])
    result_df['attribute_majority_vote'] = split[1]

    result_df = result_df.round(2)
    if output_path:
        result_df.to_csv(output_path, index=False)
    return result_df


def outliers_percentage_table(data, config, output_path=None):
    start_year = config["START_YEAR"]
    end_year = config["END_YEAR"]
    train_interval = config["DBSCAN_TRAIN_INTERVAL"]
    test_interval = config["DBSCAN_TEST_INTERVAL"]
    list_of_dfs = {0: [], 1: []}
    for train_start in range(start_year, end_year - train_interval - test_interval + 2):
        train_end = train_start + train_interval - 1
        test_start = train_end + 1
        test_end = test_start + test_interval - 1
        train_data, test_data = preprocess.cluster_frame(data, config, train_start, train_end, test_start, test_end)
        for i, data_set in enumerate([train_data, test_data]):
            grouped_df = data_set.groupby(config["SEVERE_FEATURE"]).size().reset_index(name='total_count')

            # Filter the DataFrame for rows where 'cluster' is equal to -1
            outliers = data_set[data_set['cluster'] == -1]
            print("i", i, outliers.shape[0])
            outliers_grouped = outliers.groupby(config["SEVERE_FEATURE"]).size().reset_index(name='outliers_count')

            merged_df = pd.merge(grouped_df, outliers_grouped, on=config["SEVERE_FEATURE"], how='left')
            merged_df['outliers_count'].fillna(0, inplace=True)

            merged_df['percentage_outliers'] = (merged_df['outliers_count'] / merged_df['total_count']) * 100
            list_of_dfs[i].append(merged_df)
    train_df = pd.concat(list_of_dfs[0]).groupby(config["SEVERE_FEATURE"], as_index=False).mean()
    test_df = pd.concat(list_of_dfs[1]).groupby(config["SEVERE_FEATURE"], as_index=False).mean()
    result_df = pd.merge(train_df, test_df, on=config["SEVERE_FEATURE"], suffixes=('_train', '_test'))
    if output_path:
        result_df.to_csv(output_path, index=False)
    return result_df


def location_accuracy_statistics(data, config, output_path=None):
    # THIS IS ON 20 CITIES DATA
    city_mapping = utilities.get_city_mapping(config=config)
    data = data[data[config["CITY_FEATURE"]].isin(city_mapping.keys())]
    grouped = data.groupby([config['LOCATION_ACC_FEATURE'], config['SEVERE_FEATURE']]).size().reset_index(name='count')
    grouped['normalized_count'] = grouped['count'] / grouped['count'].sum()
    if output_path:
        grouped.to_csv(output_path, index=False)
    return grouped


def theoretical_overview(config, save_path):
    data = utilities.load_data(config=config)
    data_cities = utilities.load_cities_data(config=config)
    city_mapping = utilities.get_city_mapping(config=config)
    accident_statistics_by_city_table(data=data_cities, config=config, city_mapping=city_mapping,
                                      output_path=save_path + 'accident_statistics_by_city.csv')
    accident_clusters_statistics_by_city_table(data=data_cities, config=config,
                                               output_path=save_path + 'accident_clusters_statistics_by_city.csv')
    accident_clusters_statistics_by_attribute_table(data=data_cities, config=config,
                                                    output_path=
                                                    save_path + 'accident_clusters_statistics_by_attribute.csv')
    outliers_percentage_table(data=data_cities, config=config, output_path=save_path + 'outliers_percentage.csv')

    location_accuracy_statistics(data=data, config=config, output_path=save_path + 'location_accuracy.csv')


if __name__ == '__main__':
    israel_config = utilities.load_config()
    theoretical_overview(israel_config, save_path='data/theoretical overview/israel_model_20/')

