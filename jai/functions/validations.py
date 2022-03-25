from .classes import PossibleDtypes


def hyperparams_validation(dtype: str):
    possible = []
    must = []
    if dtype == PossibleDtypes.selfsupervised:
        possible.extend([
            'batch_size', 'learning_rate', 'encoder_layer', 'decoder_layer',
            'hidden_latent_dim', 'dropout_rate', 'momentum',
            'pretraining_ratio', 'noise_level', 'check_val_every_n_epoch',
            'gradient_clip_val', 'gradient_clip_algorithm', 'min_epochs',
            'max_epochs', 'patience', 'min_delta', 'random_seed',
            'stochastic_weight_avg', 'pruning_method', 'pruning_amount',
            'training_type'
        ])
    elif dtype == PossibleDtypes.supervised:
        possible.extend([
            'batch_size', 'learning_rate', 'encoder_layer', 'decoder_layer',
            'hidden_latent_dim', 'dropout_rate', 'momentum',
            'pretraining_ratio', 'noise_level', 'check_val_every_n_epoch',
            'gradient_clip_val', 'gradient_clip_algorithm', 'min_epochs',
            'max_epochs', 'patience', 'min_delta', 'random_seed',
            'stochastic_weight_avg', 'pruning_method', 'pruning_amount'
        ])
    elif dtype == PossibleDtypes.recommendation_system:
        possible.extend([
            'batch_size', 'learning_rate', 'encoder_layer', 'decoder_layer',
            'hidden_latent_dim', 'dropout_rate', 'momentum',
            'pretraining_ratio', 'noise_level', 'check_val_every_n_epoch',
            'gradient_clip_val', 'gradient_clip_algorithm', 'min_epochs',
            'max_epochs', 'patience', 'min_delta', 'random_seed',
            'stochastic_weight_avg', 'pruning_method', 'pruning_amount'
        ])
    elif dtype == PossibleDtypes.image:
        possible.extend(['model_name', 'mode', 'resize_H', 'resize_W'])
    elif dtype == PossibleDtypes.text:
        possible.extend(['nlp_model', 'max_length'])
    elif dtype == PossibleDtypes.fasttext:
        possible.extend([
            'minn', 'maxn', 'dim', 'epoch', 'model', 'lr', 'ws', 'minCount',
            'neg', 'wordNgrams', 'loss', 'bucket', 'lrUpdateRate', 't'
        ])
    elif dtype == PossibleDtypes.edit:
        possible.extend([
            'nt', 'nr', 'nb', 'k', 'epochs', 'shuffle_seed', 'batch_size',
            'test_batch_size', 'channel', 'embed_dim', 'random_train',
            'random_append_train', 'maxl'
        ])

    return (possible, must)


def num_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['embedding_dim', 'scaler', 'fill_value'])
    return (possible, must)


def cat_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['embedding_dim', 'fill_value', 'min_freq'])
    return (possible, must)


def datetime_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['embedding_dim'])
    return (possible, must)


def features_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['embedding_dim', 'fill_value', 'min_freq'])
        must.extend(['dtype', 'scaler'])
    return (possible, must)


def pretrained_bases_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['embedding_dim', 'aggregation_method'])
        must.extend(['db_parent', 'id_name'])
    return (possible, must)


def split_process_validation(dtype: str):
    possible = []
    must = []
    if dtype in [
            PossibleDtypes.selfsupervised, PossibleDtypes.supervised,
            PossibleDtypes.recommendation_system
    ]:
        possible.extend(['type', 'split_column', 'test_size', 'gap'])
    return (possible, must)


def label_process_validation(dtype: str):
    possible = []
    must = []
    if dtype == PossibleDtypes.supervised:
        possible.extend(['regression_scaler', 'quantiles'])
        must.extend(['task', 'label_name'])
    return (possible, must)


def kwargs_possibilities(dtype: str):
    params = {
        'hyperparams': hyperparams_validation(dtype),
        'num_process': num_process_validation(dtype),
        'cat_process': cat_process_validation(dtype),
        'datetime_process': datetime_process_validation(dtype),
        'pretrained_bases': pretrained_bases_process_validation(dtype),
        'mycelia_bases': pretrained_bases_process_validation(dtype),
        'features': features_process_validation(dtype),
        'label': label_process_validation(dtype),
        'split': split_process_validation(dtype)
    }

    to_delete = []
    for item in params.items():
        if item[1] == ([], []):
            to_delete.append(item[0])

    for key in to_delete:
        del params[key]

    return params


def kwargs_validation(dtype: str, body: dict):
    params = kwargs_possibilities(dtype)
    params_keys = set(params.keys())
    body_keys = set(body.keys()) - set(["callback_url", "overwrite"])
    correct_used_keys = body_keys & params_keys
    incorrect_used_keys = body_keys - params_keys

    if not incorrect_used_keys:
        for key in correct_used_keys:
            if key in set(['mycelia_bases', 'pretrained_bases']):
                if isinstance(body[key], list):
                    pb_keys = [list(x.keys()) for x in body[key]]
                    used_subkeys = set().union(*pb_keys)
                else:
                    raise TypeError(
                        "'pretrained_bases' parameter must be a list of dictonaries."
                    )
            else:
                used_subkeys = set(body[key])
            possible_subkeys = set(params[key][0])
            must_subkeys = set(params[key][1])
            must_and_pos_subkeys = must_subkeys | possible_subkeys

            if not must_subkeys <= used_subkeys:
                diff = must_subkeys - used_subkeys
                raise ValueError(
                    f'{list(diff)} parameter is required for the dtype "{dtype}".'
                )
            if not used_subkeys <= must_and_pos_subkeys:
                diff = used_subkeys - must_and_pos_subkeys
                raise ValueError(
                    f'Inserted key argument(s) {list(diff)} are not a valid one for dtpe="{dtype}".'\
                        f' Please check the documentation and try again.'
                )
    else:
        raise ValueError(
            f'Inserted key argument(s) {list(incorrect_used_keys)} are not a valid one for dtype="{dtype}". '\
                 f'Please check the documentation and try again.')

    return "All inserted parameters are correct."
