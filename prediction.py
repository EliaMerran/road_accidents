import config
import preprocess
import utilities

if __name__ == '__main__':
    # Load data
    data = utilities.get_cities_data()
    # Preprocess data
    data = preprocess.preprocess(data, start_year=config.START_YEAR, end_year=config.END_YEAR,
                                 train_interval=config.TRAIN_INTERVAL,
                                 save_path='data/processed data/all_cluster_attempt.csv')
    # Split data

    # Fit model

    # Evaluate model

    # Plot results

    # Save results

# TODO:
#  - Add to cluster.csv the polygon of the cluster
#  - Finish the theoretical overview
#  - decide on train, test, validation split
#  - maybe make the preprocess on 2008-2021
# remove data that the x y is not certain
