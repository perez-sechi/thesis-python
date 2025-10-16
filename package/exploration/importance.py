import numpy as np
import pandas as pd


def shap_ranked_top_features(shap_values, X):
    '''
    Rank features by their importance
    :param shap_values: shap values
    :param X: pandas DataFrame
    :return: pandas DataFrame
    '''
    mean_abs_values = np.abs(shap_values).mean(0)
    feature_names = X.columns

    shap_importance = pd.DataFrame(
        list(zip(feature_names, mean_abs_values)),
        columns=['col_name', 'mean_abs_importance']
    )
    shap_importance.sort_values(
        by=['mean_abs_importance'],
        ascending=False, inplace=True
    )
    shap_importance['rank'] = range(1, 1 + len(shap_importance))
    cols = shap_importance.columns.tolist()
    cols = cols[-1:] + cols[:-1]

    return shap_importance[cols]


def shap_ranked_top_interactions(shap_interaction_values, X, head=15):
    '''
    Rank features by their importance
    :param shap_interaction_values: shap_interaction_values
    :param X: pandas DataFrame
    :return: pandas DataFrame
    '''
    minus_mean_abs_values = -np.abs(shap_interaction_values).mean(0)
    indices = np.dstack(np.unravel_index(np.argsort(
        minus_mean_abs_values.ravel()), minus_mean_abs_values.shape))[0]

    result = pd.DataFrame({
        'feature1': [],
        'feature2': [],
        'mean_interaction': [],
        'mean_abs_interaction': []
    })

    count = 0
    for i, (row, col) in enumerate(indices):
        if count >= head:
            break
        if row == col:
            continue
        if len(result[
            (result['feature1'] == X.columns[col])
                & (result['feature2'] == X.columns[row])
        ]) > 0:
            continue
        result.loc[i] = {
            'feature1': X.columns[row],
            'feature2': X.columns[col],
            'mean_interaction': shap_interaction_values[:, row, col].mean(),
            'mean_abs_interaction': np.abs(shap_interaction_values[:, row, col]).mean()
        }
        count += 1

    return result


def shap_top_important_features(shap_values, X, head=15):
    '''
    Get top important features
    :param shap_values: shap values
    :param X: pandas DataFrame
    :param head: int
    :return: list
    '''
    shap_importance = shap_ranked_top_features(shap_values, X)

    return list(shap_importance.head(head)["col_name"])


def shap_top_important_interactions(shap_interaction_values, X, head=15):
    '''
    Get top important interactions
    :param shap_interaction_values: shap shap_interaction_values
    :param X: pandas DataFrame
    :param head: int
    :return: list
    '''
    return shap_ranked_top_interactions(shap_interaction_values, X, head)


def shap_median_important_features(shap_values, X, elem=15):
    '''
    Get median important features
    :param shap_values: shap values
    :param X: pandas DataFrame
    '''
    shap_importance = shap_ranked_top_features(
        shap_values, X
    )
    median = shap_importance["rank"].median()
    top_elem = elem // 2
    remaining_elem = elem % 2
    least_elem = top_elem + 1 if remaining_elem > 0 else 0
    top_median_importance = shap_importance[
        shap_importance["rank"] >= median
    ].head(least_elem)
    least_median_importance = shap_importance[
        shap_importance["rank"] < median
    ].tail(top_elem)

    return \
        list(top_median_importance["col_name"]) \
        + list(least_median_importance["col_name"])
