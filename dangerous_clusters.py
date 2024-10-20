import utilities
import preprocess
import json
import numpy as np
import xgboost as xgb
import geopandas as gpd
import pandas as pd
import webbrowser
import os
from shapely import wkt


def top_dangerous_city_clusters_on_map(config, n_clusters, threshold=0):
    model_path = config['CITY_MODEL_PATH']
    config_path = model_path.replace(".json", "_configuration.json")
    with open(config_path) as f:
        model_config = json.load(f)
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model(model_path)
    model_clusters = pd.read_csv(model_config["CITY_CLUSTERS_DATA_PATH"])
    X_train, y_train, X_test, y_test = utilities.split_train_test(model_clusters, model_config)
    y_prob = xgb_classifier.predict_proba(X_test)

    # Create a DataFrame with predicted probabilities and geometries
    result_df = model_clusters.loc[X_test.index]
    result_df['y_prob'] = y_prob[:, 1]
    y_pred = y_prob[:, 1] > threshold
    result_df['y_pred'] = y_pred
    result_df['accurate'] = result_df['y_pred'] == result_df['dangerous']
    # fix cluster name
    result_df['cluster'] = result_df['cluster'].astype(str)
    result_df['cluster'] = result_df['cluster'].apply(lambda x: x.replace('2021', ''))
    #####
    # result_df = result_df[result_df['type'] == 'Severe Turnaround']

    # Sort DataFrame by predicted probabilities in descending order
    result_df = result_df.sort_values(by='y_prob', ascending=False)
    # Get the top 5 clusters
    top_clusters = result_df.head(n_clusters)
    # hover_columns = (top_clusters.drop(columns=['cluster','geometry_x','train_index', 'geometry_y', 'test_index']).
    #                  columns.tolist())
    hover_columns = ['cluster', 'size', 'train_severe', 'test_severe', 'type', 'y_prob', 'y_pred','dangerous', 'accurate']
    top_clusters['geometry_x'] = top_clusters['geometry_x'].apply(wkt.loads)
    top_clusters.drop(columns=['train_index', 'test_index'], inplace=True)
    top_gdf = gpd.GeoDataFrame(top_clusters, geometry='geometry_x', crs=config["CRS"])
    top_gdf = top_gdf.explode(index_parts=True)
    out = top_gdf.explore(marker_kwds={"radius": 8}, column='cluster', # symbol='accurate',
                          legend=True, cmap='tab20',
                          tooltip=hover_columns)

    path = os.path.realpath(f"out/graphs_dangerous_clusters_top_{n_clusters}_clusters.html")
    out.save(path)

    webbrowser.open('file://' + path)


if __name__ == '__main__':
    israel_config = utilities.load_config()
    top_dangerous_city_clusters_on_map(israel_config, 1000)
