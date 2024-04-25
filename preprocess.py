import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

import config
import utilities


def preprocess(data, start_year=config.START_YEAR, end_year=config.END_YEAR, train_interval=config.TRAIN_INTERVAL,
               test_interval = config.TEST_INTERVAL, save_path=None, min_cluster_size=config.MIN_CLUSTER_SIZE,
               cluster_eps=config.EPS, test_eps=config.EPS, min_samp=config.MIN_SAMPLES ):
    list_of_dfs = []
    # i = 0
    for train_start in range(start_year, end_year - train_interval - test_interval + 2):
        # i = i + 1
        # print(i)
        train_end = train_start + train_interval - 1
        test_start = train_end + 1
        test_end = test_start + test_interval - 1
        train_data, test_data = cluster_frame(data, train_start, train_end, test_start, test_end, cluster_eps, test_eps,
                                              min_samp)
        train = gpd.GeoDataFrame(train_data["X Y cluster".split()],
                                 geometry=gpd.points_from_xy(train_data.X, train_data.Y, crs="EPSG:2039"))
        df_clusters = train_data.groupby(['cluster']).size().reset_index(name='size')
        # add polygon of train accidents in cluster
        df_clusters = df_clusters.merge(
            train.dissolve(by='cluster').drop(columns=['X', 'Y']),
            how='left',
            on='cluster'
        )
        # add index of train accidents in cluster
        train_indices = train_data.groupby('cluster').apply(lambda group: group.index.tolist())
        df_clusters['train_index'] = df_clusters['cluster'].map(train_indices)

        # add polygon of test accidents in cluster
        test = gpd.GeoDataFrame(test_data["X Y cluster".split()],
                                 geometry=gpd.points_from_xy(test_data.X, test_data.Y, crs="EPSG:2039"))
        df_clusters = df_clusters.merge(
            test.dissolve(by='cluster').drop(columns=['X', 'Y']),
            how='left',
            on='cluster'
        )
        # add index of test accidents in cluster
        test_indices = test_data.groupby('cluster').apply(lambda group: group.index.tolist())
        df_clusters['test_index'] = df_clusters['cluster'].apply(lambda cluster: test_indices.get(cluster, []))
        # df_clusters['test_index'] = df_clusters['cluster'].map(test_indices)

        df_clusters = df_clusters[df_clusters['size'] >= min_cluster_size]
        df_clusters = add_severe_count(df_clusters, train_data, 'train_severe')
        df_clusters = add_severe_count(df_clusters, test_data, 'test_severe')
        df_clusters = df_clusters[df_clusters.cluster != -1]
        df_clusters = add_cluster_type(df_clusters)
        for attribute in config.ATTRIBUTES_DICT.keys():
            attribute_df = get_clusters_by_attribute(train_data, test_data, attribute,
                                                     config.ATTRIBUTES_DICT[attribute])
            attribute_df = attribute_df[['cluster', f'{attribute}_majority_vote']]
            df_clusters = pd.merge(df_clusters, attribute_df, on='cluster', how='outer')
        df_clusters['cluster'] = str(test_start) + df_clusters['cluster'].astype(str)
        df_clusters['test_year'] = test_start
        df_clusters['dangerous'] = df_clusters['test_severe'] > 0
        list_of_dfs.append(df_clusters)
    result_df = pd.concat(list_of_dfs, ignore_index=True)
    # # Extract text features
    # cats = result_df.select_dtypes(exclude=np.number).columns.tolist()
    #
    # # Convert to Pandas category
    # for col in cats:
    #     result_df[col] = result_df[col].astype('category')
    if save_path is not None:
        result_df.to_csv(save_path, index=False)
    return result_df


def cluster_frame(frame, train_start, train_end, test_start, test_end, cluster_eps=config.EPS, test_eps=config.EPS,
                  min_samp=config.MIN_SAMPLES):
    train_data = frame[frame.SHNAT_TEUNA.between(train_start, train_end)].copy()
    test_data = frame[frame.SHNAT_TEUNA.between(test_start, test_end)].copy()

    clustering = DBSCAN(eps=cluster_eps, min_samples=min_samp).fit(train_data["X Y".split()])
    train_data["cluster"] = clustering.labels_
    train_clusters = train_data[train_data["cluster"] >= 0]  # un-clustered points are -1

    # Spatial join of test data to assign clusters to test data
    test = gpd.GeoDataFrame(test_data["X Y".split()],
                            geometry=gpd.points_from_xy(test_data.X, test_data.Y, crs="EPSG:2039"))
    train = gpd.GeoDataFrame(train_clusters["X Y cluster".split()],
                             geometry=gpd.points_from_xy(train_clusters.X, train_clusters.Y, crs="EPSG:2039"))
    test = gpd.sjoin_nearest(test, train, how="left", distance_col="dist", lsuffix="test", rsuffix="train")
    test = test.reset_index().drop_duplicates(subset="pk_teuna_fikt").set_index("pk_teuna_fikt")

    # assert test["dist"].max() <= test_eps
    # change cluster to -1 if dist > test_eps
    test.loc[test["dist"] > test_eps, "cluster"] = -1
    test_data["cluster"] = test["cluster"]

    return train_data, test_data

# For all points as clusters
# def cluster_frame(frame, train_start, train_end, test_start, test_end, cluster_eps=config.EPS, test_eps=config.EPS,
#                   min_samp=config.MIN_SAMPLES):
#     train_data = frame[frame.SHNAT_TEUNA.between(train_start, train_end)].copy()
#     test_data = frame[frame.SHNAT_TEUNA.between(test_start, test_end)].copy()
#
#     train_data["cluster"] = list(range(len(train_data)))
#     train_clusters = train_data[train_data["cluster"] >= 0]  # un-clustered points are -1
#
#     # Spatial join of test data to assign clusters to test data
#     test = gpd.GeoDataFrame(test_data["X Y".split()],
#                             geometry=gpd.points_from_xy(test_data.X, test_data.Y, crs="EPSG:2039"))
#     train = gpd.GeoDataFrame(train_clusters["X Y cluster".split()],
#                              geometry=gpd.points_from_xy(train_clusters.X, train_clusters.Y, crs="EPSG:2039"))
#     test = gpd.sjoin_nearest(test, train, how="left", distance_col="dist", lsuffix="test", rsuffix="train",
#                              max_distance=test_eps)
#     test = test.reset_index().drop_duplicates(subset="pk_teuna_fikt").set_index("pk_teuna_fikt")
#
#     # assert test["dist"].max() <= test_eps
#
#     # change cluster to -1 if dist > test_eps
#     test.loc[test["dist"] > test_eps, "cluster"] = -1
#     test_data["cluster"] = test["cluster"]
#
#     return train_data, test_data


def get_clusters_by_attribute(train_frame, test_frame, attribute, attribute_dict,
                              min_cluster_size=config.MIN_CLUSTER_SIZE):
    df_counts = train_frame.groupby(['cluster', attribute]).size().reset_index(name='count')
    df_clusters = df_counts.pivot(index='cluster', columns=attribute, values='count').fillna(0).astype(int)
    df_clusters.reset_index(inplace=True)
    df_clusters.columns.name = None
    df_clusters['cluster_size'] = df_clusters.drop('cluster', axis=1).sum(axis=1)
    df_clusters = df_clusters[df_clusters['cluster_size'] >= min_cluster_size]
    df_clusters = add_severe_count(df_clusters, train_frame, 'train_severe')
    df_clusters = add_severe_count(df_clusters, test_frame, 'test_severe')

    df_clusters = add_cluster_type(df_clusters)

    # Rename the columns
    df_clusters.rename(columns=attribute_dict, inplace=True)
    filtered_dict = {key: value for key, value in attribute_dict.items() if value != 'unknown'}

    # Create a new column 'traffic_light_majority_vote'
    attribute_columns = [col for col in filtered_dict.values() if col in df_clusters.columns]
    if len(attribute_columns):
        df_clusters[f'{attribute}_majority_vote'] = df_clusters[attribute_columns].idxmax(axis=1)
        # Set 'unknown' if the maximum value is 0
        mask_zero_max = df_clusters[attribute_columns].max(axis=1) == 0
        df_clusters.loc[mask_zero_max, f'{attribute}_majority_vote'] = 'unknown'
    else:
        df_clusters[f'{attribute}_majority_vote'] = 'unknown'
    df_clusters = df_clusters[df_clusters.cluster != -1]
    return df_clusters


def add_severe_count(df_clusters, frame, column_name):
    df_clusters = df_clusters.merge(
        frame[frame['HUMRAT_TEUNA'] < 3].groupby('cluster').size().reset_index(name=column_name),
        how='left',
        on='cluster'
    )
    df_clusters[column_name].fillna(0, inplace=True)
    df_clusters[column_name] = df_clusters[column_name].astype(int)
    return df_clusters


def add_cluster_type(df_clusters):
    # Define the conditions
    conditions = [
        (df_clusters['train_severe'] == 0) & (df_clusters['test_severe'] > 0),
        (df_clusters['train_severe'] == 0) & (df_clusters['test_severe'] == 0),
        (df_clusters['train_severe'] > 0) & (df_clusters['test_severe'] > 0),
        (df_clusters['train_severe'] > 0) & (df_clusters['test_severe'] == 0)
    ]

    # Define the corresponding values for each condition
    values = ['Severe Turnaround', 'Stable Safety', 'Consistent Severity', 'Improved Safety']

    # Create the 'type' column based on the conditions
    df_clusters['type'] = np.select(conditions, values, default='Undefined')
    return df_clusters


if __name__ == '__main__':
    # Load data
    data = utilities.get_cities_data()
    # Preprocess data
    data = preprocess(data, start_year=config.START_YEAR, end_year=config.END_YEAR,
                      train_interval=config.TRAIN_INTERVAL,
                      save_path=f'data/processed data/clusters_{config.START_YEAR}_{config.END_YEAR}.csv')

