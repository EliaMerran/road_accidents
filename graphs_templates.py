import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.subplots as sp


def save_plot(fig, save_path, title):
    modified_title = ''.join(c if c.isalpha() else '_' for c in title)
    fig.write_image(save_path + modified_title + '.png')
    fig.write_html(save_path + modified_title + '.html')


def plot_roc_curve(df, color_feature, title='ROC Curve', show=True, save_path=None):
    fig = px.line(df, x='fpr', y='tpr', title=title,
                  color=color_feature, hover_data=['thresholds', 'auc'], labels={'fpr': 'FPR', 'tpr': 'TPR'})
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig


def plot_top_prediction_labels(X_test, y_test, xgb_classifier, threshold, minor_only=False,
                               title='Top Predictions',show=True, save_path=None):
    X = X_test.copy()
    y = y_test.copy()
    if minor_only:
        X = X_test[X_test['train_severe'] == 0].copy()
        y = y_test[X_test['train_severe'] == 0].copy()
    y_prob = xgb_classifier.predict_proba(X)
    y_pred = y_prob[:, 1] > threshold
    tp_lst, fp_lst, tn_lst, fn_lst = [], [], [], []
    # my_range = range(10, len(y), 10)
    my_range = range(10, 100)
    for top in my_range:
        df_temp = pd.DataFrame({'y_prob': y_prob[:, 1], 'y_test': y, 'y_pred': y_pred})
        df_top = (df_temp.sort_values(by='y_prob', ascending=False)).head(top)
        conf_matrix = confusion_matrix(df_top['y_test'], df_top['y_pred'])
        # Check the shape of the confusion matrix
        if conf_matrix.shape == (1, 1):
            tp = conf_matrix[0][0]
            tn = fp = fn = 0
        else:
            tn, fp, fn, tp = conf_matrix.ravel()

        tp_lst.append(tp)
        fp_lst.append(fp)
        tn_lst.append(tn)
        fn_lst.append(fn)

    df_top_plot = pd.DataFrame(
        {'top': my_range, 'tp': tp_lst, 'fp': fp_lst, 'tn': tn_lst, 'fn': fn_lst})
    fig = px.line(df_top_plot, x='top', y=['tp', 'fp', 'tn', 'fn'],
                  labels={'value': 'Count', 'variable': 'Metric'},
                  title=title,
                  width=800, height=500)
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig


def plot_multi_top_prediction_labels(figs, titles, title='Top Predictions', show=True, save_path=None):
    n_figs = len(figs)
    fig = sp.make_subplots(rows=1, cols=n_figs, subplot_titles=titles, shared_xaxes=True, shared_yaxes=True)
    for i in range(n_figs):
        fig.add_trace(figs[i]['data'][0], row=1, col=i + 1)
        fig.add_trace(figs[i]['data'][1], row=1, col=i + 1)
        fig.add_trace(figs[i]['data'][2], row=1, col=i + 1)
        fig.add_trace(figs[i]['data'][3], row=1, col=i + 1)
    fig.update_layout(title_text=title)
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig


def plot_cluster_type_dist(df, title='Cluster Type Distribution', show=True, save_path=None):
    fig = px.pie(df, names='type', title=title)
    fig.update_layout(
        width=600,  # Width of the figure in pixels
        height=600  # Height of the figure in pixels
    )
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig


def plot_multi_cluster_type_dist(figs, titles, title='Multi Cluster Type Distribution',show=True, save_path=None):
    fig = sp.make_subplots(rows=1, cols=len(figs), subplot_titles=titles, specs=[[{'type': 'pie'}] * len(figs)])
    # Add each pie chart to the subplot figure
    for i, data in enumerate(figs):
        fig.add_trace(data.data[0], row=1, col=i + 1)
    fig.update_layout(title_text='Accidents Type Distribution in Test Set')
    fig.update_layout(
        width=1000,  # Width of the figure in pixels
        height=500  # Height of the figure in pixels
    )
    fig.update_layout(title_text=title)
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig


def plot_accidents_by_attr_per_year(df, attribute, attribute_year_name, title='Accidents per Year', show=True,
                                    save_path=None):
    grouped = df.groupby([attribute_year_name, attribute]).size().reset_index(name='count')
    fig = px.line(grouped, x=attribute_year_name, y='count', color=attribute, title=title,
                  labels={attribute_year_name: 'Year'}, markers=True)
    if show:
        fig.show()
    if save_path:
        save_plot(fig, save_path, title)
    return fig
