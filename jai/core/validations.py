import warnings
from typing import List

import numpy as np
import pandas as pd
from pydantic import ValidationError, parse_obj_as

from ..types.generic import PossibleDtypes
from .exceptions import DeprecatedError, ParamError


def check_response(
    model, obj, list_of: bool = False, as_list: bool = False, as_dict: bool = False
):
    """
    Checks if response from API follows the expected structure.

    Args:
        model (_type_): expected structure for the response.
        obj (_type_): the response from API.
        list_of (bool, optional): If the obj follows the structure of
        a list of model defined in `model`. Defaults to False.
        as_list (bool, optional): If the obj follows the structure of
        a list. Defaults to False.
        as_dict (bool, optional): If the obj follows the structure of
        a dict. Defaults to False.

    Raises:
        ValueError: if more than one of `list_of`, `as_list` and `as_dict`
        parameters are are set to true.
        ValueError: If the response does not correspond to the expected structure

    Returns:
        The response values as expected.
    """
    if sum([list_of, as_list, as_dict]) > 1:
        raise ValueError("Can't use `list_of`, `as_list` and `as_dict` simultaneously.")

    if model is None:
        warnings.warn(
            "No check is available for this method when `safe_mode` is on.",
            stacklevel=3,
        )
        return obj

    try:
        if list_of:
            return [i.dict() for i in parse_obj_as(List[model], obj)]
        elif as_list:
            return [i.dict() for i in parse_obj_as(model, obj)]
        elif as_dict:
            return {k: v.dict() for k, v in parse_obj_as(model, obj).items()}
        return parse_obj_as(model, obj)

    except ValidationError:
        raise ValueError(
            "Validation Failed. This error occurred because `safe_mode=True`."
            "The API may have changed, please try updating your version of jai-sdk."
            "If the error persists, please report the error on an issue so we can work on a fix."
        )


def check_dtype_and_clean(data, db_type):
    """
    Check data type and remove NAs from the data.
    This is a protected method.

    Args
    ----
    data : pandas.DataFrame or pandas.Series
        Data to be checked and cleaned.

    db_type : str
        Database type (Supervised, SelfSupervised, Text...)

    Return
    ------
    data : pandas.DataFrame or pandas.Series
        Data without NAs
    """
    # TODO: improve this function
    if not isinstance(data, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError(
            f"Inserted data is of type `{data.__class__.__name__}`,"
            f"but supported types are np.ndarray, pandas.Series or pandas.DataFrame"
        )

    if isinstance(data, np.ndarray):
        if not data.any():
            raise ValueError(f"Inserted data is empty.")
        elif data.ndim == 1:
            data = pd.Series(data)
        elif data.ndim == 2:
            data = pd.DataFrame(data)
        else:
            raise ValueError(
                f"Inserted 'np.ndarray' data has many dimensions ({data.ndim}). JAI only accepts up to 2-d inputs."
            )

    if (
        db_type
        in [
            PossibleDtypes.text,
            PossibleDtypes.fasttext,
            PossibleDtypes.edit,
            PossibleDtypes.vector,
        ]
        and data.isna().to_numpy().any()
    ):
        warnings.warn(f"Droping NA values.")
        data = data.dropna()
    return data


# Helper function to validate name lengths before training
def check_name_lengths(name, cols):
    invalid_cols = []
    for col in cols:
        if len(name + "_" + col) > 32:
            invalid_cols.append(col)

    if len(invalid_cols):
        raise ValueError(
            f"The following column names are too large to concatenate\
            with database '{name}':\n{invalid_cols}\nPlease enter a shorter database name or\
            shorter column names; 'name_column' string must be at most 32 characters long."
        )


def hyperparams_validation(dtype: str):
    possible = []
    must = []
    if dtype == PossibleDtypes.selfsupervised:
        possible.extend(
            [
                "batch_size",
                "learning_rate",
                "encoder_layer",
                "decoder_layer",
                "hidden_latent_dim",
                "dropout_rate",
                "momentum",
                "pretraining_ratio",
                "noise_level",
                "check_val_every_n_epoch",
                "gradient_clip_val",
                "gradient_clip_algorithm",
                "min_epochs",
                "max_epochs",
                "patience",
                "min_delta",
                "random_seed",
                "swa_parameters",
                "pruning_method",
                "pruning_amount",
                "training_type",
            ]
        )
    elif dtype == PossibleDtypes.supervised:
        possible.extend(
            [
                "batch_size",
                "learning_rate",
                "encoder_layer",
                "decoder_layer",
                "hidden_latent_dim",
                "dropout_rate",
                "momentum",
                "pretraining_ratio",
                "noise_level",
                "check_val_every_n_epoch",
                "gradient_clip_val",
                "gradient_clip_algorithm",
                "min_epochs",
                "max_epochs",
                "patience",
                "min_delta",
                "random_seed",
                "swa_parameters",
                "pruning_method",
                "pruning_amount",
            ]
        )
    elif dtype == PossibleDtypes.recommendation_system:
        possible.extend(
            [
                "batch_size",
                "learning_rate",
                "encoder_layer",
                "decoder_layer",
                "hidden_latent_dim",
                "dropout_rate",
                "momentum",
                "pretraining_ratio",
                "noise_level",
                "check_val_every_n_epoch",
                "gradient_clip_val",
                "gradient_clip_algorithm",
                "min_epochs",
                "max_epochs",
                "patience",
                "min_delta",
                "random_seed",
                "swa_parameters",
                "pruning_method",
                "pruning_amount",
            ]
        )
    elif dtype == PossibleDtypes.image:
        possible.extend(["model_name", "mode", "resize_H", "resize_W"])
    elif dtype == PossibleDtypes.text:
        possible.extend(["nlp_model", "max_length"])
    elif dtype == PossibleDtypes.fasttext:
        possible.extend(
            [
                "minn",
                "maxn",
                "dim",
                "epoch",
                "model",
                "lr",
                "ws",
                "minCount",
                "neg",
                "wordNgrams",
                "loss",
                "bucket",
                "lrUpdateRate",
                "t",
            ]
        )
    elif dtype == PossibleDtypes.edit:
        possible.extend(
            [
                "nt",
                "nr",
                "nb",
                "k",
                "epochs",
                "shuffle_seed",
                "batch_size",
                "test_batch_size",
                "channel",
                "embed_dim",
                "random_train",
                "random_append_train",
                "maxl",
            ]
        )

    return (possible, must)


def num_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["embedding_dim", "scaler", "fill_value"])
    return (possible, must)


def cat_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["embedding_dim", "fill_value", "min_freq"])
    return (possible, must)


def datetime_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["embedding_dim"])
    return (possible, must)


def features_process_validation(dtype: str):
    possible = []
    must = ["dtype"]
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["scaler", "embedding_dim", "fill_value", "min_freq"])
    return (possible, must)


def pretrained_bases_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["embedding_dim", "aggregation_method"])
        must.extend(["db_parent", "id_name"])
    return (possible, must)


def split_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
        PossibleDtypes.selfsupervised,
        PossibleDtypes.supervised,
        PossibleDtypes.recommendation_system,
    ]:
        possible.extend(["type", "split_column", "test_size", "gap"])
    return (possible, must)


def label_process_validation(dtype: str):
    possible = []
    must = []
    if dtype == PossibleDtypes.supervised:
        possible.extend(["regression_scaler", "quantiles"])
        must.extend(["task", "label_name"])
    return (possible, must)


def kwargs_possibilities(dtype: str):
    params = {
        "hyperparams": hyperparams_validation(dtype),
        "num_process": num_process_validation(dtype),
        "cat_process": cat_process_validation(dtype),
        "datetime_process": datetime_process_validation(dtype),
        "pretrained_bases": pretrained_bases_process_validation(dtype),
        "mycelia_bases": pretrained_bases_process_validation(dtype),
        "features": features_process_validation(dtype),
        "label": label_process_validation(dtype),
        "split": split_process_validation(dtype),
    }

    to_delete = []
    for item in params.items():
        if item[1] == ([], []):
            to_delete.append(item[0])

    for key in to_delete:
        del params[key]

    return params


def plurality(list_keys):
    if len(list_keys) > 1:
        return f"arguments `{'`, `'.join(list_keys)}` are"
    return f"argument `{list(list_keys)[0]}` is"


def kwargs_validation(db_type: str, **kwargs):
    doc_msg = "Please check the documentation and try again."
    params = kwargs_possibilities(db_type)
    params_keys = set(params.keys())
    body_keys = set(kwargs.keys()) - set(["callback_url", "overwrite"])
    correct_used_keys = body_keys & params_keys
    incorrect_used_keys = body_keys - params_keys

    if incorrect_used_keys:
        raise ParamError(
            f'Inserted {plurality(incorrect_used_keys)} not a valid one for dtype="{db_type}". {doc_msg}'
        )

    if "label" not in body_keys and db_type == PossibleDtypes.supervised:
        raise ParamError(f"Missing the required arguments: `label`. {doc_msg}")

    body = {"db_type": db_type, "callback_url": kwargs.get("callback_url", None)}
    for key in correct_used_keys:
        if key == "mycelia_bases":
            raise DeprecatedError(
                "`mycelia_bases` has been deprecated, please use `pretrained_bases` instead."
            )
        elif key == "pretrained_bases":
            if not isinstance(kwargs[key], list):
                raise TypeError(
                    "'pretrained_bases' parameter must be a list of dictonaries."
                )
            pb_keys = [list(x.keys()) for x in kwargs[key]]
            used_subkeys = set().union(*pb_keys)
        elif key == "features":
            if not isinstance(kwargs[key], dict):
                raise TypeError(
                    "'features' parameter must be a dictonary of dictonaries."
                )
            pb_keys = [list(x.keys()) for x in kwargs[key].values()]
            used_subkeys = set().union(*pb_keys)
        else:
            used_subkeys = set(kwargs[key])
        possible_subkeys = set(params[key][0])
        must_subkeys = set(params[key][1])
        must_and_pos_subkeys = must_subkeys | possible_subkeys

        if not must_subkeys <= used_subkeys:
            diff = must_subkeys - used_subkeys
            raise ParamError(
                f'{list(diff)} parameter is required for the dtype "{db_type}".'
            )
        if not used_subkeys <= must_and_pos_subkeys:
            diff = used_subkeys - must_and_pos_subkeys
            raise ParamError(
                f'Inserted {plurality(diff)} not a valid one for dtpe="{db_type}". {doc_msg}'
            )
        body[key] = kwargs[key]

    if body.get("hyperparams", {}).get("patience", 10) > 0:
        print("Training might finish early due to early stopping criteria.")
    return body
