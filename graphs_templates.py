import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import plotly.subplots as sp
import plotly.graph_objects as go


def plot_roc_curve(df, title, color_feature, show=True, save_path=None):
    fig = px.line(df, x='fpr', y='tpr', title=f'ROC Curve for {title}',
                  color=color_feature, hover_data=['thresholds'], labels={'fpr': 'FPR', 'tpr': 'TPR'})
    # fig.update_layout(
    #     width=600,
    #     height=600
    # )
    if show:
        fig.show()
    if save_path:
        modified_title = ''.join(c if c.isalpha() else '_' for c in title)
        fig.write_image(save_path + f'roc_curve_{modified_title}.png')
    return fig


def plot_top_prediction_labels(X_test, y_test, xgb_classifier, threshold, minor_only=False,
                               title='', show=True, save_path=None):
    X = X_test.copy()
    y = y_test.copy()
    if minor_only:
        X = X_test[X_test['train_severe'] == 0].copy()
        y = y_test[X_test['train_severe'] == 0].copy()
    y_prob = xgb_classifier.predict_proba(X)
    y_pred = y_prob[:, 1] > threshold
    tp_lst,fp_lst, tn_lst, fn_lst= [], [], [], []
    my_range = range(10, 101)
    for top in my_range:
        df_temp = pd.DataFrame({'y_prob': y_prob[:, 1], 'y_test': y, 'y_pred': y_pred})
        df_top = (df_temp.sort_values(by='y_prob', ascending=False)).head(top)
        tn, fp, fn, tp = confusion_matrix(df_top['y_test'], df_top['y_pred']).ravel()
        tp_lst.append(tp)
        fp_lst.append(fp)
        tn_lst.append(tn)
        fn_lst.append(fn)

    df_top_plot = pd.DataFrame(
        {'top': my_range, 'tp': tp_lst, 'fp': fp_lst, 'tn': tn_lst, 'fn': fn_lst})
    fig = px.line(df_top_plot, x='top', y=['tp', 'fp', 'tn', 'fn'],
                  labels={'value': 'Count', 'variable': 'Metric'},
                  title=f'Top of Predictions with Threshold {threshold} {title}',
                  width=800, height=500)
    if show:
        fig.show()
    if save_path:
        fig.write_image(save_path + f'Top of Predictions with Threshold {threshold}.png')
    return fig


def plot_multi_top_prediction_labels(figs, titles, title='', show=True, save_path=None):
    n_figs = len(figs)
    # Create subplots
    subplot = sp.make_subplots(rows=1, cols=n_figs, subplot_titles=titles, shared_xaxes=True, shared_yaxes=True)

    for i in range(n_figs):
        subplot.add_trace(figs[i]['data'][0], row=1, col=i + 1)
        subplot.add_trace(figs[i]['data'][1], row=1, col=i + 1)
        subplot.add_trace(figs[i]['data'][2], row=1, col=i + 1)
        subplot.add_trace(figs[i]['data'][3], row=1, col=i + 1)

    # Update layout
    subplot.update_layout(title_text=f'Top of Predictions {title}')
    # subplot.update_layout(
    #     width=1000,  # Width of the figure in pixels
    #     height=500  # Height of the figure in pixels
    # )

    if show:
        subplot.show()
    if save_path:
        subplot.write_image(save_path + 'tp_fp_vs_top_min_sample.png')
    return subplot


def plot_cluster_type_dist(df,title, show=True, save_path=None):
    fig = px.pie(df, names='type', title=title)
    fig.update_layout(
        width=600,  # Width of the figure in pixels
        height=600  # Height of the figure in pixels
    )
    if show:
        fig.show()
    if save_path:
        fig.write_image(save_path + f'cluster_type_dist_{title}.png')
    return fig


def plot_multi_cluster_type_dist(figs, titles, show=True, save_path=None):
    subplot = sp.make_subplots(rows=1, cols=len(figs), subplot_titles=titles, specs=[[{'type': 'pie'}] * len(figs)])
    # Add each pie chart to the subplot figure
    for i, data in enumerate(figs):
        subplot.add_trace(data.data[0], row=1, col=i + 1)
    subplot.update_layout(title_text='Accidents Type Distribution in Test Set')
    subplot.update_layout(
        width=1000,  # Width of the figure in pixels
        height=500  # Height of the figure in pixels
    )
    if show:
        subplot.show()
    if save_path:
        subplot.write_image(f'graphs/adaptive_dbscan/cluster_type_dist.png')
    return subplot