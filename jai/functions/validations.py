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
    if (dtype == PossibleDtypes.selfsupervised) or (
            dtype == PossibleDtypes.supervised):
        possible.extend(['embedding_dim', 'scaler', 'fill_value'])
    return (possible, must)


def cat_process_validation(dtype: str):
    possible = []
    must = []
    if (dtype == PossibleDtypes.selfsupervised) or (
            dtype == PossibleDtypes.supervised):
        possible.extend(['embedding_dim', 'fill_value', 'min_freq'])
    return (possible, must)


def datetime_process_validation(dtype: str):
    possible = []
    must = []
    if (dtype == PossibleDtypes.selfsupervised) or (
            dtype == PossibleDtypes.supervised):
        possible.extend(['embedding_dim'])
    return (possible, must)


def features_process_validation(dtype: str):
    possible = []
    must = []
    if (dtype == PossibleDtypes.selfsupervised) or (
            dtype == PossibleDtypes.supervised):
        possible.extend(['embedding_dim', 'fill_value', 'min_freq'])
        must.extend(['dtype', 'scaler'])
    return (possible, must)


def pretrained_bases_process_validation(dtype: str):
    possible = []
    must = []
    if dtype == PossibleDtypes.supervised:
        possible.extend(['embedding_dim', 'aggregation_method'])
        must.extend(['db_parent', 'id_name'])
    elif dtype == PossibleDtypes.selfsupervised:
        possible.extend(['embedding_dim'])
        must.extend(['db_parent', 'id_name'])
    return (possible, must)


def split_process_validation(dtype: str):
    possible = []
    must = []
    if (dtype == PossibleDtypes.selfsupervised) or (
            dtype == PossibleDtypes.supervised):
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
        'features': features_process_validation(dtype),
        'label': label_process_validation(dtype),
        'split': split_process_validation(dtype)
    }
    return params


def kwargs_validation(dtype: str, body: dict):
    params = kwargs_possibilities(dtype)
    for key in body.keys():
        if key in params.keys():
            for subkey in params[key][1]:
                if subkey not in body[key].keys():
                    raise ValueError(
                        f'{subkey} parameter is required for the dtype {dtype}.'
                    )
            for subkey in body[key]:
                if (subkey not in params[key][0]) and (subkey
                                                       not in params[key][1]):
                    raise ValueError(
                        f'Inserted key argument "{subkey}" is not a valid one for dtype="{dtype}". Please check the documentation and try again.'
                    )
        return "All inserted parameters are correct."


if __name__ == '__main__':

    TEST = {'model_name': 'mnasnet', 'mode': 'classifier'}

    print(kwargs_validation('Image', TEST))