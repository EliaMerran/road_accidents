import geopandas as gpd
from sklearn.cluster import DBSCAN
import plotly.express as px

import utilities
import webbrowser
import os
import pandas as pd
from typing import Callable, Any
import folium


def plot_dbscan_clusters_in_cities(config, data, city_mapping,):
    eps = config['EPS']
    min_sample = config["MIN_SAMPLES"]
    year = config['START_YEAR']
    # Plot for each city
    for city_code, city_name in city_mapping.items():
        df_city = data[data[config['CITY_FEATURE']] == city_code]
        df_city = df_city[df_city[config['YEAR_FEATURE']].between(year, year+config['DBSCAN_TRAIN_INTERVAL']-1)]
        clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(df_city["X Y".split()])
        df_city["cluster"] = clustering.labels_
        df_city = df_city[df_city["cluster"] >= 0]  # un-clustered points are -1
        df_city['cluster'] = df_city['cluster'].astype('category')
        color_order = df_city.cluster.unique().sort_values()
        # Create an interactive Plotly scatter plot
        fig = px.scatter(df_city, x='X', y='Y', color='cluster', hover_data=config['YEAR_FEATURE'],
                         title=f'Clusters for {city_name}, eps = {eps}, min_sample = {min_sample}',
                         category_orders={'cluster': color_order})

        df_city = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city.X, df_city.Y, crs=config['CRS']))

        out = df_city.explore(marker_kwds={"radius": 5}, column='cluster')

        out.save(f"graphs/dbscan_clusters/israel/{city_name}.html")

        webbrowser.open('file://' + os.path.realpath(f"graphs/dbscan_clusters/israel/{city_name}.html"))



def calculate_param_from_accidents_density(data: pd.DataFrame, cities_attributes: pd.DataFrame, city_name: str,
                                           city_code: str, formula: Callable[..., int], *args: Any) -> int:
    x = calculate_accidents_density(data, cities_attributes, city_name, city_code)
    # Apply the given function to calculate the result
    return formula(*args, x)


def calculate_accidents_density(data, cities_attributes, city_name, city_code):
    # Filter DataFrame based on the given city
    city_data = data[data['SEMEL_YISHUV'] == city_code]
    accidents_average = city_data['SHNAT_TEUNA'].value_counts().mean() * 5
    area = cities_attributes[cities_attributes['Name'] == city_name]['Area,km2'].values[0] * 10
    accidents_density = accidents_average / area
    return accidents_density


if __name__ == '__main__':
    # Load config and data
    city_israel_config = utilities.load_config()
    cities_data = pd.read_csv(city_israel_config["CITY_DATA_PATH"], index_col=city_israel_config['INDEX_FEATURE'])
    city_mapping = {5000: "Tel Aviv - Yafo"}
    # city_mapping = utilities.get_city_mapping()
    plot_dbscan_clusters_in_cities(city_israel_config, cities_data, city_mapping)




