import os.path

import graphs_templates
import utilities
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def performance_overview(model_path, data, clusters, config, save_path, title, show=True):
    clusters_data = clusters.copy()
    graphs_templates.plot_accidents_by_attr_per_year(data, config["SEVERE_FEATURE"], config["YEAR_FEATURE"],
                                                     title=f'Accidents by Severity per Year {title}',
                                                     show=show,
                                                     save_path=save_path)
    graphs_templates.plot_accidents_by_attr_per_year(clusters_data, 'type', 'test_year',
                                                     title=f'Cluster Type Distribution per Year {title}',
                                                     show=show,
                                                     save_path=save_path)

    X_train, y_train, X_test, y_test = utilities.split_train_test(clusters_data, config)

    # Load model from file
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model(model_path)
    pred_fig = graphs_templates.plot_top_prediction_labels(X_test, y_test, xgb_classifier, 0, show=show,
                                                           save_path=save_path, title=f'Top Predictions {title}')
    y_prob = xgb_classifier.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    # Calculate AUC
    auc = roc_auc_score(y_test, y_prob[:, 1])
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
                       "model": title, 'auc': auc})
    graphs_templates.plot_roc_curve(df, title=f'ROC Curve {title}', color_feature='model', show=show,
                                    save_path=save_path)
    return pred_fig, df


def compare_models(models, configs, clusters, titles, save_path=None):
    pred_figs = []
    roc_dfs = []
    for model, conf, cluster_data, title in zip(models, configs, clusters, titles):
        cities_data = utilities.load_cities_data(conf)
        fig, roc_df = performance_overview(model, cities_data, cluster_data, conf, save_path, title=title, show=False)
        pred_figs.append(fig)
        roc_dfs.append(roc_df)
    graphs_templates.plot_multi_top_prediction_labels(pred_figs, titles=titles,
                                                      title='Top Predictions', show=True, save_path=save_path)
    roc = pd.concat(roc_dfs, ignore_index=True)
    print(roc.groupby('model').mean())
    graphs_templates.plot_roc_curve(roc, title=f'ROC Curve', color_feature='model', show=True,
                                    save_path=save_path)


def compare_israel_uk(save_path):
    config_israel = utilities.load_config()
    config_uk = utilities.load_config(use_uk=True)
    configs = [config_israel, config_uk]
    pred_figs = []
    roc_dfs = []
    for conf in configs:
        data = utilities.load_data(conf)
        cities_data = utilities.load_cities_data(conf)
        for i in range(2):
            if i == 0:
                new_save_path = save_path + conf["COUNTRY"] + '_'
                model_path = conf["MODEL_PATH"]
                city = False
            else:
                new_save_path = save_path + conf["COUNTRY"] + '_Cities_'
                model_path = conf["CITY_MODEL_PATH"]
                city = True
            fig, roc_df = performance_overview(model_path, data, cities_data, conf, new_save_path, city)
            pred_figs.append(fig)
            roc_dfs.append(roc_df)
    graphs_templates.plot_multi_top_prediction_labels(pred_figs, titles=[f'Israel', 'Israel Cities', 'UK', 'UK Cities'],
                                                      title='Top Predictions', show=True, save_path=save_path)
    roc = pd.concat(roc_dfs, ignore_index=True)
    graphs_templates.plot_roc_curve(roc, title=f'ROC Curve', color_feature='country', show=True,
                                    save_path=save_path)


if __name__ == '__main__':
    models = ['models/israel_city_model.json',
              'models/uk_city_model.json']
    clusters = []
    israel_config = utilities.load_config()
    uk_config = utilities.load_config(use_uk=True)
    configs = [israel_config, uk_config]
    for model in models:
        df = pd.read_csv(model.replace(".json", "_data.csv"))
        clusters.append(df)
    titles = ['Israel City Model', 'United Kingdom City Model']
    compare_models(models, configs, clusters, titles, save_path=os.path.realpath("out/fig_"))
