import numpy as np
import warnings

from copy import deepcopy
from tqdm import tqdm
from .functions.utils_funcs import multikeysort

__all__ = [
    "find_threshold", "process_similar", "process_predict",
    "process_resolution"
]


def find_threshold(results, sample_size=0.1, quantile=0.05):
    """
    Auxiliar function to find a threshold value.

    Takes a sample of size **sample_size** of the **results** list and uses the
    **quantile** of the distances of the sample to use as threshold.

    This is a automated function, we strongly advise to set the threshold manualy
    to get more accurate results.

    Parameters
    ----------
    results : list of dicts, output of similar
        DESCRIPTION.
    sample_size : float, optional
        Percentage of the results taken to calculate the threshold. If
        **len(results)** is too small, i.e., **len(results) * sample_size** is
        less than 1, then we use **sample_size=0.5** or 1. The default is 0.1.
    quantile : float, optional
        Quantile of the distances of all the query results of the sample taken.
        We suggest to use the similar method with a top_k big enough for the quantile,
        i.e., the total number of distances is `len(results) * sample_size * top_k`,
        top_k helps to get more values of distances as of using a small top_k
        will make a distance group of only distances close to 0 and threshold
        may not be representative. The default is 0.05.

    Returns
    -------
    float
        Threshold result.

    """
    results = deepcopy(results)
    if len(results) <= 1 // sample_size:
        n = len(results) // 2
    else:
        n = int(len(results) * sample_size)
    n = max(n, 1)

    samples = np.random.randint(0, len(results), n)
    distribution = []
    for s in tqdm(samples, desc="Fiding threshold"):
        d = [l['distance'] for l in results[s]['results'][1:]]
        distribution.extend(d)
    threshold = np.quantile(distribution, quantile)
    warnings.warn("Threshold calculated automatically.")
    print(f"\nrandom sample size: {n}\nthreshold: {threshold}\n")
    return threshold


def process_similar(results,
                    threshold: float = None,
                    return_self: bool = True,
                    skip_null: bool = True):
    """
    Process the output from the similar methods.

    For each of the inputs, gives back the closest value. If result_self is False,
    avoids returning cases where 'id' is equal to 'query_id' and returns the
    next closest if necessary.

    Parameters
    ----------
    results : List of Dicts.
        output from similar methods.
    threshold : float, optional
        value for the distance threshold. The default is None.
        if set to None, we used the auxiliar function find_threshold.
    return_self : bool, optional
        option to return the queried id from the query result or not. The default is True.
    skip_null: bool, optional
        option to skip ids without similar results, if False, returns empty results.
        The default is True.

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
        threshold = find_threshold(results)

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


def process_resolution(results,
                       threshold=None,
                       return_self=True,
                       res_id="resolution_id"):
    """
    Process the results of similarity for resolution goals.

    Differs from process_similiar on cases where A is similar to B and B is
    similar to C, it should give the result of both A and B are similar to C,
    and so on.

    Parameters
    ----------
    results : List of Dicts.
        output from similar methods.
    threshold : float, optional
        value for the distance threshold. The default is None.
        if set to None, we used the auxiliar function find_threshold.
    return_self : bool, optional
        option to return the queried id from the query result or not. The default is True.
    res_id: str, optional
        name of the key for the resolution. The default is "resolution_id".

    Returns
    -------
    connect : list of dicts
        List of dicts with each id and their correspondent resolution.

    """

    results = deepcopy(results)
    results = sorted(results, key=lambda x: x['query_id'])
    if threshold is None:
        threshold = find_threshold(results)

    # The if A is similar to B and B is similar to C, then C should be A
    connect = []  # all connected relations
    con_aux = {}  # past relationships
    for q in tqdm(results, desc='Process'):
        qid = q['query_id']
        if qid in con_aux.keys():
            qid = con_aux[qid]
        res = multikeysort(q['results'], ['distance', 'id'])
        filt = filter(lambda x: x['distance'] <= threshold, res)
        for item in filt:
            _id = item['id']
            # if id hasn't been solved
            if _id not in con_aux.keys():
                # if is itself (root solution)
                if _id == qid:
                    con_aux[_id] = qid
                    if return_self:
                        temp = {"id": _id, res_id: qid}
                        connect.append(temp)
                # if not itself or distance less than threshold
                else:
                    temp = {"id": _id, res_id: qid}
                    con_aux[_id] = qid
                    connect.append(temp)
            # if id has been solved, keep going
            else:
                continue
    return connect
