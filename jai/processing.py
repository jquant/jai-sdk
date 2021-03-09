import numpy as np

from copy import deepcopy
from tqdm import tqdm
from .functions.utils_funcs import multikeysort

__all__ = ["process_similar", "process_predict"]


def process_similar(results,
                    threshold: float = None,
                    return_self: bool = True,
                    skip_null: bool = True):
    """
    Process the output from the similar methods.

    Parameters
    ----------
    results : List of Dicts.
        output from similar methods.
    threshold : float, optional
        value for the distance threshold. The default is None.
        if set to None, takes a random 1% of the results and uses the 10%
        quantile of the distances distributions as the threshold.
    return_self : bool, optional
        option to return the queried id from the query result or not. The default is True.
    skip_null: bool, optional
        option to skip ids without similar results. The default is True.

    Raises
    ------
    NotImplementedError
        If priority inputed is not implemented.

    Returns
    -------
    list
        mapping the query id to the similar value.

    """
    results = deepcopy(results)
    if threshold is None:
        samples = np.random.randint(0, len(results), len(results) // (100))
        distribution = []
        for s in tqdm(samples, desc="Fiding threshold"):
            d = [l['distance'] for l in results[s]['results'][1:]]
            distribution.extend(d)
        threshold = np.quantile(distribution, .1)
    print(f"threshold: {threshold}\n")

    similar = []
    for q in tqdm(results, desc='Process'):
        sort = multikeysort(q['results'], ['distance', 'id'])
        zero, one = sort[0], sort[1]
        if zero['distance'] <= threshold and (zero['id'] != q['query_id']
                                              or return_self):
            zero['query_id'] = q['query_id']
            similar.append(zero)
        elif one['distance'] <= threshold:
            one['query_id'] = q['query_id']
            similar.append(one)
        elif not skip_null:
            mock = {"query_id": q['query_id'], "id": None, "distance": None}
            similar.append(mock)
        else:
            continue
    return similar


def process_predict(predicts):
    """
    Process the output from the predict methods from supervised models.

    Parameters
    ----------
    predicts : List of Dicts.
        output from predict methods.

    Raises
    ------
    NotImplementedError
        If unexpected predict type. {type(example)}

    Returns
    -------
    list
        mapping the query id to the predicted value.

    """
    predicts = deepcopy(predicts)
    example = predicts[0]['predict']
    if isinstance(example, dict):
        predict_proba = True
    elif isinstance(example, str):
        predict_proba = False
    else:
        raise ValueError(f"Unexpected predict type. {type(example)}")

    sanity_check = []
    for query in tqdm(predicts, desc='Predict all ids'):
        if predict_proba == False:
            sanity_check.append(query)
        else:
            predict = max(query['predict'], key=query['predict'].get)
            confidence_level = round(query['predict'][predict] * 100, 2)
            sanity_check.append({
                'id': query['id'],
                'predict': predict,
                'probability(%)': confidence_level
            })
    return sanity_check
