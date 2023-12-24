import pandas as pd
import numpy as np
import geopandas as gpd

import utilities
import config
import preprocess


# creates a table of the following format:
# city, population, total accidents, (of them) property damage, light, severe, casualty
def accident_statistics_by_city_table(data, city_mapping, output_path=None):
    # Create city mapping
    city_info = utilities.get_city_info(sort_by_population=True).head(config.NUM_CITIES)
    # Group and pivot data
    pivot = (
        data.groupby(['SEMEL_YISHUV', 'SHNAT_TEUNA', 'HUMRAT_TEUNA'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Mean over years
    pivot = pivot.groupby('SEMEL_YISHUV').mean().reset_index()
    # Map city names and calculate total accidents
    pivot['city'] = pivot['SEMEL_YISHUV'].map(city_mapping)
    pivot['total_accidents'] = pivot[[1, 2, 3, 4]].sum(axis=1)
    pivot['population'] = pivot['SEMEL_YISHUV'].map(city_info.set_index('סמל יישוב')['סך הכל אוכלוסייה 2021'])

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
def accident_clusters_statistics_by_city_table(data, city_mapping, output_path=None):
    attribute = 'RAMZOR'
    list_of_dfs = []
    for city_code, city_name in city_mapping.items():
        city_data = data[data['SEMEL_YISHUV'] == city_code].copy()
        mean_accidents = city_data.groupby('SHNAT_TEUNA').size().mean()
        mean_minor_accidents = city_data[city_data['HUMRAT_TEUNA'] >= 3].groupby('SHNAT_TEUNA').size().mean()
        mean_severe_accidents = city_data[city_data['HUMRAT_TEUNA'] < 3].groupby('SHNAT_TEUNA').size().mean()

        num_clusters = []
        cluster_size = []
        min_cluster_count = []
        average_cluster_count = []
        max_cluster_count = []
        std_cluster_count = []
        df_final = pd.DataFrame()
        for train_start in range(config.START_YEAR, config.END_TRAIN_YEAR):
            train_end = train_start + 2
            test_start = train_end + 1
            test_end = test_start
            city_train, city_test = preprocess.cluster_frame(city_data, train_start=train_start, train_end=train_end,
                                                             test_start=test_start, test_end=test_end)
            city_train = city_train[city_train.cluster != -1]
            clusters = city_train.groupby('cluster').size().reset_index(name='count')
            clusters = clusters[clusters['count'] >= config.MIN_CLUSTER_SIZE]
            # clusters = clusters[clusters['cluster'] != -1]
            num_clusters.append(len(clusters))
            describe = clusters['count'].describe()
            min_cluster_count.append(describe['min'])
            average_cluster_count.append(describe['mean'])
            max_cluster_count.append(describe['max'])
            std_cluster_count.append(describe['std'])

            train = gpd.GeoDataFrame(city_train["X Y cluster".split()],
                                     geometry=gpd.points_from_xy(city_train.X, city_train.Y, crs="EPSG:2039"))
            cluster_size.append(train.dissolve(by='cluster').minimum_bounding_radius().mean())
            res = preprocess.get_clusters_by_attribute(city_train, city_test, attribute,
                                                       config.ATTRIBUTES_DICT[attribute], config.MIN_CLUSTER_SIZE)
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
def accident_clusters_statistics_by_attribute_table(data, output_path=None):
    list_of_dfs = []
    num_clusters_lst = []
    for train_start in range(config.START_YEAR, config.END_TRAIN_YEAR):
        train_end = train_start + 2
        test_start = train_end + 1
        test_end = test_start
        city_train, city_test = preprocess.cluster_frame(data, train_start=train_start, train_end=train_end,
                                                         test_start=test_start, test_end=test_end)
        clusters = city_train.groupby('cluster').size().reset_index(name='size')
        clusters = clusters[clusters['size'] >= config.MIN_CLUSTER_SIZE]
        n_cluster = len(clusters[clusters['cluster'] != -1])
        num_clusters_lst.append(n_cluster)
        for attribute in config.ATTRIBUTES_DICT.keys():
            res = preprocess.get_clusters_by_attribute(city_train, city_test, attribute,
                                                       config.ATTRIBUTES_DICT[attribute], config.MIN_CLUSTER_SIZE)
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
            df_grouped['min_cluster_size'] = config.MIN_CLUSTER_SIZE
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


def outliers_percentage_table(data, output_path=None):
    list_of_dfs = []
    for train_start in range(config.START_YEAR, config.END_YEAR - config.TRAIN_INTERVAL + 1):
        train_end = train_start + config.TRAIN_INTERVAL - 1
        test_start = train_end + 1
        test_end = test_start
        train_data, test_data = preprocess.cluster_frame(data, train_start, train_end, test_start, test_end)

        # Group by 'HUMRAT_TEUNA' and count the number of occurrences for each group
        grouped_df = train_data.groupby('HUMRAT_TEUNA').size().reset_index(name='total_count')

        # Filter the DataFrame for rows where 'cluster' is equal to -1
        outliers = train_data[train_data['cluster'] == -1]

        # Group by 'HUMRAT_TEUNA' in the cluster -1 DataFrame and count the number of occurrences for each group
        outliers_grouped = outliers.groupby('HUMRAT_TEUNA').size().reset_index(name='outliers_count')

        # Merge the two DataFrames based on 'HUMRAT_TEUNA'
        merged_df = pd.merge(grouped_df, outliers_grouped, on='HUMRAT_TEUNA', how='left')

        # Fill NaN values with 0 (for cases where there are no accidents in cluster -1 for a specific 'HUMRAT_TEUNA')
        merged_df['outliers_count'].fillna(0, inplace=True)

        # Calculate the percentage for each 'HUMRAT_TEUNA'
        merged_df['percentage_outliers'] = (merged_df['outliers_count'] / merged_df['total_count']) * 100
        list_of_dfs.append(merged_df)
    result_df = pd.concat(list_of_dfs).groupby('HUMRAT_TEUNA', as_index=False).mean()
    if output_path:
        result_df.to_csv(output_path, index=False)
    return result_df


def location_accuracy_statistics(data, city_mapping, output_path=None):
    data = data[data['SEMEL_YISHUV'].isin(city_mapping.keys())]
    grouped = data.groupby(['STATUS_IGUN','HUMRAT_TEUNA']).size().reset_index(name='count')
    grouped['normalized_count'] = grouped['count'] / grouped['count'].sum()
    if output_path:
        grouped.to_csv(output_path, index=False)
    return grouped


if __name__ == '__main__':
    data_cities = utilities.get_cities_data()
    city_mapping = utilities.get_city_mapping()
    accident_statistics_by_city_table(data_cities, city_mapping,
                                      output_path='data/theoretical overview/accident_statistics_by_city.csv')
    accident_clusters_statistics_by_city_table(data_cities, city_mapping,
                                                       output_path='data/theoretical overview/'
                                                                   'accident_clusters_statistics_by_city.csv')
    accident_clusters_statistics_by_attribute_table(data_cities,
                                                    output_path='data/theoretical overview/'
                                                                'accident_clusters_statistics_by_attribute.csv')
    outliers_percentage_table(data_cities, output_path='data/theoretical overview/outliers_percentage.csv')
    data = utilities.get_accidents_data()
    location_accuracy_statistics(data,city_mapping, output_path='data/theoretical overview/location_accuracy.csv')
    # data_cities = pd.read_csv('data/processed data/cities_accidents.csv', index_col='pk_teuna_fikt')
    # accident_clusters_statistics_by_city_table(data_cities, city_mapping, output_path='try.csv')
