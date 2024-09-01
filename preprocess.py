import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd

# import config
import utilities


def preprocess(data, config, save_path=None):
    start_year = config["START_YEAR"]
    end_year = config["END_YEAR"]
    DBSCAN_TRAIN_INTERVAL = config["DBSCAN_TRAIN_INTERVAL"]
    DBSCAN_TEST_INTERVAL = config["DBSCAN_TEST_INTERVAL"]
    min_cluster_size = config["MIN_CLUSTER_SIZE"]
    crs = config["CRS"]
    list_of_dfs = []
    i = 0
    for train_start in range(start_year, end_year - DBSCAN_TRAIN_INTERVAL - DBSCAN_TEST_INTERVAL + 2):
        i = i + 1
        print(i, train_start)
        train_end = train_start + DBSCAN_TRAIN_INTERVAL - 1
        test_start = train_end + 1
        test_end = test_start + DBSCAN_TEST_INTERVAL - 1
        train_data, test_data = cluster_frame(data, config, train_start, train_end, test_start, test_end)
        train = gpd.GeoDataFrame(train_data[[config["X_FEATURE"], config["Y_FEATURE"], 'cluster']],
                                 geometry=gpd.points_from_xy(train_data[config["X_FEATURE"]],
                                                             train_data[config["Y_FEATURE"]], crs=crs))
        df_clusters = train_data.groupby(['cluster']).size().reset_index(name='size')
        # add polygon of train accidents in cluster
        df_clusters = df_clusters.merge(
            train.dissolve(by='cluster').drop(columns=[config["X_FEATURE"], config["Y_FEATURE"]]),
            how='left',
            on='cluster'
        )
        # add index of train accidents in cluster
        train_indices = train_data.groupby('cluster').apply(lambda group: group.index.tolist())
        df_clusters['train_index'] = df_clusters['cluster'].map(train_indices)

        # add polygon of test accidents in cluster
        test = gpd.GeoDataFrame(test_data[[config["X_FEATURE"], config["Y_FEATURE"], 'cluster']],
                                geometry=gpd.points_from_xy(test_data[config["X_FEATURE"]],
                                                            test_data[config["Y_FEATURE"]], crs=crs))
        df_clusters = df_clusters.merge(
            test.dissolve(by='cluster').drop(columns=[config["X_FEATURE"], config["Y_FEATURE"]]),
            how='left',
            on='cluster'
        )
        # add index of test accidents in cluster
        test_indices = test_data.groupby('cluster').apply(lambda group: group.index.tolist())
        df_clusters['test_index'] = df_clusters['cluster'].apply(lambda cluster: test_indices.get(cluster, []))
        df_clusters = df_clusters[df_clusters['size'] >= min_cluster_size]
        df_clusters = add_severe_count(df_clusters, config, train_data, 'train_severe')
        df_clusters = add_severe_count(df_clusters, config, test_data, 'test_severe')

        # NEW
        # add severe and minor by years
        df_clusters = add_severe_years_count(df_clusters, config, train_data, 'train_severe')
        # df_clusters = add_minor_years_count(df_clusters, config, train_data, 'train_minor')
        # NEW 2
        # df_clusters = add_severity_years_count(df_clusters, config, train_data, 'train')
        ## END NEW

        df_clusters = df_clusters[df_clusters.cluster != -1]
        df_clusters = add_cluster_type(df_clusters)
        for attribute in config["ATTRIBUTE_DICT"].keys():
            attribute_df = get_clusters_by_attribute(train_data, test_data, attribute,
                                                     config["ATTRIBUTE_DICT"][attribute], config)
            attribute_df = attribute_df[['cluster', f'{attribute}_majority_vote']]
            df_clusters = pd.merge(df_clusters, attribute_df, on='cluster', how='outer')
        df_clusters['cluster'] = str(test_start) + df_clusters['cluster'].astype(str)
        df_clusters['test_year'] = test_start
        df_clusters['dangerous'] = df_clusters['test_severe'] > 0
        list_of_dfs.append(df_clusters)
    result_df = pd.concat(list_of_dfs, ignore_index=True)
    if save_path is not None:
        result_df.to_csv(save_path, index=False)
    return result_df


def cluster_frame(frame, config, train_start, train_end, test_start, test_end):
    cluster_eps = config["EPS"]
    test_eps = config["EPS"]
    min_samp = config["MIN_SAMPLES"]
    crs = config["CRS"]
    train_data = frame[frame[config["YEAR_FEATURE"]].between(train_start, train_end)].copy()
    test_data = frame[frame[config["YEAR_FEATURE"]].between(test_start, test_end)].copy()

    clustering = DBSCAN(eps=cluster_eps, min_samples=min_samp).fit(
        train_data[[config["X_FEATURE"], config["Y_FEATURE"]]])
    train_data["cluster"] = clustering.labels_
    train_clusters = train_data[train_data["cluster"] >= 0]  # un-clustered points are -1

    # Spatial join of test data to assign clusters to test data
    test = gpd.GeoDataFrame(test_data[[config["X_FEATURE"], config["Y_FEATURE"]]],
                            geometry=gpd.points_from_xy(test_data[config["X_FEATURE"]], test_data[config["Y_FEATURE"]],
                                                        crs=crs))
    train = gpd.GeoDataFrame(train_clusters[[config["X_FEATURE"], config["Y_FEATURE"], "cluster"]],
                             geometry=gpd.points_from_xy(train_clusters[config["X_FEATURE"]],
                                                         train_clusters[config["Y_FEATURE"]], crs=crs))
    test = gpd.sjoin_nearest(test, train, how="left", distance_col="dist", lsuffix="test", rsuffix="train")
    test = test.reset_index().drop_duplicates(subset=config["INDEX_FEATURE"]).set_index(config["INDEX_FEATURE"])

    # assert test["dist"].max() <= test_eps
    # change cluster to -1 if dist > test_eps
    test.loc[test["dist"] > test_eps, "cluster"] = -1
    test_data["cluster"] = test["cluster"]

    return train_data, test_data


def get_clusters_by_attribute(train_frame, test_frame, attribute, attribute_dict,
                              config):
    min_cluster_size = config["MIN_CLUSTER_SIZE"]
    df_counts = train_frame.groupby(['cluster', attribute]).size().reset_index(name='count')
    df_clusters = df_counts.pivot(index='cluster', columns=attribute, values='count').fillna(0).astype(int)
    df_clusters.reset_index(inplace=True)
    df_clusters.columns.name = None
    df_clusters['cluster_size'] = df_clusters.drop('cluster', axis=1).sum(axis=1)
    df_clusters = df_clusters[df_clusters['cluster_size'] >= min_cluster_size]
    df_clusters = add_severe_count(df_clusters, config, train_frame, 'train_severe')
    df_clusters = add_severe_count(df_clusters, config, test_frame, 'test_severe')

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


def add_severe_count(df_clusters, config, frame, column_name):
    df_clusters = df_clusters.merge(
        frame[frame[config["SEVERE_FEATURE"]] < 3].groupby('cluster').size().reset_index(name=column_name),
        how='left',
        on='cluster'
    )
    df_clusters[column_name].fillna(0, inplace=True)
    df_clusters[column_name] = df_clusters[column_name].astype(int)
    return df_clusters


def add_minor_count(df_clusters, config, frame, column_name):
    df_clusters = df_clusters.merge(
        frame[frame[config["SEVERE_FEATURE"]] >= 3].groupby('cluster').size().reset_index(name=column_name),
        how='left',
        on='cluster'
    )
    df_clusters[column_name].fillna(0, inplace=True)
    df_clusters[column_name] = df_clusters[column_name].astype(int)
    return df_clusters


def add_severe_years_count(df_clusters, config, frame, column_name):
    for i, year in enumerate(frame[config["YEAR_FEATURE"]].unique()):
        df_clusters = df_clusters.merge(
            frame[(frame[config["SEVERE_FEATURE"]] < 3) & (frame[config["YEAR_FEATURE"]] == year)].groupby('cluster')
            .size().reset_index(name=f'{column_name}_{i}'),
            how='left',
            on='cluster'
        )
        df_clusters[f'{column_name}_{i}'].fillna(0, inplace=True)
        df_clusters[f'{column_name}_{i}'] = df_clusters[f'{column_name}_{i}'].astype(int)
    return df_clusters


def add_minor_years_count(df_clusters, config, frame, column_name):
    for i, year in enumerate(frame[config["YEAR_FEATURE"]].unique()):
        df_clusters = df_clusters.merge(
            frame[(frame[config["SEVERE_FEATURE"]] >= 3) & (frame[config["YEAR_FEATURE"]] == year)].groupby('cluster')
            .size().reset_index(name=f'{column_name}_{i}'),
            how='left',
            on='cluster'
        )
        df_clusters[f'{column_name}_{i}'].fillna(0, inplace=True)
        df_clusters[f'{column_name}_{i}'] = df_clusters[f'{column_name}_{i}'].astype(int)
    return df_clusters


def add_severity_years_count(df_clusters, config, frame, column_name):
    for i, year in enumerate(frame[config["YEAR_FEATURE"]].unique()):
        for severity in frame[config["SEVERE_FEATURE"]].unique():
            df_clusters = df_clusters.merge(
                frame[(frame[config["SEVERE_FEATURE"]] == severity) & (frame[config["YEAR_FEATURE"]] == year)].
                groupby('cluster')
                .size().reset_index(name=f'{column_name}_{year}_{severity}'),
                how='left',
                on='cluster'
            )
            df_clusters[f'{column_name}_{year}_{severity}'].fillna(0, inplace=True)
            df_clusters[f'{column_name}_{year}_{severity}'] = (df_clusters[f'{column_name}_{year}_{severity}'].
                                                               astype(int))
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
    # # Load config
    # config = utilities.load_config(use_uk=True)
    # # Load data
    # data_israel = pd.read_csv(config["DATA_PATH"], index_col=config["INDEX_FEATURE"])
    # data_israel = data_israel[data_israel.STATUS_IGUN == 1]
    # attribute_dict = utilities.create_attribute_dict(config["COUNTRY"], config["CODEBOOK_PATH"])
    # cluster = preprocess(data_israel, config, attribute_dict)
    # print(cluster.head())

    # Load config
    config_my = utilities.load_config(use_uk=True)
    # Load data
    data_uk, attribute_dict_my = utilities.load_data(config_my)
    clusters = preprocess(data_uk, config_my, attribute_dict_my)
    print(clusters.head())
