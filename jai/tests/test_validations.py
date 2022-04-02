from jai.core.types import PossibleDtypes
from jai.core.validations import (
    num_process_validation, cat_process_validation,
    datetime_process_validation, features_process_validation,
    pretrained_bases_process_validation, split_process_validation,
    label_process_validation, kwargs_possibilities, kwargs_validation,
    hyperparams_validation)
from jai.core.exceptions import DeprecatedError
import pytest
import warnings


@pytest.mark.parametrize('dtype, results', [
    (PossibleDtypes.selfsupervised, ([
        'batch_size', 'learning_rate', 'encoder_layer', 'decoder_layer',
        'hidden_latent_dim', 'dropout_rate', 'momentum', 'pretraining_ratio',
        'noise_level', 'check_val_every_n_epoch', 'gradient_clip_val',
        'gradient_clip_algorithm', 'min_epochs', 'max_epochs', 'patience',
        'min_delta', 'random_seed', 'stochastic_weight_avg', 'pruning_method',
        'pruning_amount', 'training_type'
    ], [])),
    (PossibleDtypes.supervised, ([
        'batch_size', 'learning_rate', 'encoder_layer', 'decoder_layer',
        'hidden_latent_dim', 'dropout_rate', 'momentum', 'pretraining_ratio',
        'noise_level', 'check_val_every_n_epoch', 'gradient_clip_val',
        'gradient_clip_algorithm', 'min_epochs', 'max_epochs', 'patience',
        'min_delta', 'random_seed', 'stochastic_weight_avg', 'pruning_method',
        'pruning_amount'
    ], [])),
    (PossibleDtypes.image,
     (['model_name', 'mode', 'resize_H', 'resize_W'], [])),
    (PossibleDtypes.text, (['nlp_model', 'max_length'], [])),
    (PossibleDtypes.fasttext, ([
        'minn', 'maxn', 'dim', 'epoch', 'model', 'lr', 'ws', 'minCount', 'neg',
        'wordNgrams', 'loss', 'bucket', 'lrUpdateRate', 't'
    ], [])),
    (PossibleDtypes.edit, ([
        'nt', 'nr', 'nb', 'k', 'epochs', 'shuffle_seed', 'batch_size',
        'test_batch_size', 'channel', 'embed_dim', 'random_train',
        'random_append_train', 'maxl'
    ], []))
])
def test_hyperparams_validation(dtype, results):
    assert hyperparams_validation(dtype) == results


@pytest.mark.parametrize('dtype, results',
                         [(PossibleDtypes.selfsupervised,
                           (['embedding_dim', 'scaler', 'fill_value'], [])),
                          (PossibleDtypes.supervised,
                           (['embedding_dim', 'scaler', 'fill_value'], [])),
                          (PossibleDtypes.image, ([], [])),
                          (PossibleDtypes.text, ([], [])),
                          (PossibleDtypes.fasttext, ([], [])),
                          (PossibleDtypes.edit, ([], []))])
def test_num_process_validation(dtype, results):
    assert num_process_validation(dtype) == results


@pytest.mark.parametrize('dtype, results',
                         [(PossibleDtypes.selfsupervised,
                           (['embedding_dim', 'fill_value', 'min_freq'], [])),
                          (PossibleDtypes.supervised,
                           (['embedding_dim', 'fill_value', 'min_freq'], [])),
                          (PossibleDtypes.image, ([], [])),
                          (PossibleDtypes.text, ([], [])),
                          (PossibleDtypes.fasttext, ([], [])),
                          (PossibleDtypes.edit, ([], []))])
def test_cat_process_validation(dtype, results):
    assert cat_process_validation(dtype) == results


@pytest.mark.parametrize('dtype, results',
                         [(PossibleDtypes.selfsupervised,
                           (['embedding_dim'], [])),
                          (PossibleDtypes.supervised, (['embedding_dim'], [])),
                          (PossibleDtypes.image, ([], [])),
                          (PossibleDtypes.text, ([], [])),
                          (PossibleDtypes.fasttext, ([], [])),
                          (PossibleDtypes.edit, ([], []))])
def test_datetime_process_validation(dtype, results):
    assert datetime_process_validation(dtype) == results


@pytest.mark.parametrize(
    'dtype, results',
    [(PossibleDtypes.selfsupervised,
      (['embedding_dim', 'fill_value', 'min_freq'], ['dtype', 'scaler'])),
     (PossibleDtypes.supervised,
      (['embedding_dim', 'fill_value', 'min_freq'], ['dtype', 'scaler'])),
     (PossibleDtypes.image, ([], [])), (PossibleDtypes.text, ([], [])),
     (PossibleDtypes.fasttext, ([], [])), (PossibleDtypes.edit, ([], []))])
def test_features_process_validation(dtype, results):
    assert features_process_validation(dtype) == results


@pytest.mark.parametrize(
    'dtype, results',
    [(PossibleDtypes.selfsupervised,
      (['embedding_dim', 'aggregation_method'], ['db_parent', 'id_name'])),
     (PossibleDtypes.supervised,
      (['embedding_dim', 'aggregation_method'], ['db_parent', 'id_name'])),
     (PossibleDtypes.image, ([], [])), (PossibleDtypes.text, ([], [])),
     (PossibleDtypes.fasttext, ([], [])), (PossibleDtypes.edit, ([], []))])
def test_pretrained_bases_process_validation(dtype, results):
    assert pretrained_bases_process_validation(dtype) == results


@pytest.mark.parametrize('dtype, results',
                         [(PossibleDtypes.selfsupervised,
                           (['type', 'split_column', 'test_size', 'gap'], [])),
                          (PossibleDtypes.supervised,
                           (['type', 'split_column', 'test_size', 'gap'], [])),
                          (PossibleDtypes.image, ([], [])),
                          (PossibleDtypes.text, ([], [])),
                          (PossibleDtypes.fasttext, ([], [])),
                          (PossibleDtypes.edit, ([], []))])
def test_split_process_validation(dtype, results):
    assert split_process_validation(dtype) == results


@pytest.mark.parametrize(
    'dtype, results',
    [(PossibleDtypes.selfsupervised, ([], [])),
     (PossibleDtypes.supervised,
      (['regression_scaler', 'quantiles'], ['task', 'label_name'])),
     (PossibleDtypes.image, ([], [])), (PossibleDtypes.text, ([], [])),
     (PossibleDtypes.fasttext, ([], [])), (PossibleDtypes.edit, ([], []))])
def test_label_process_validation(dtype, results):
    assert label_process_validation(dtype) == results


@pytest.mark.parametrize(
    'dtype, keys',
    [(PossibleDtypes.selfsupervised, [
        'hyperparams', 'num_process', 'cat_process', 'datetime_process',
        'pretrained_bases', 'mycelia_bases', 'features', 'split'
    ]),
     (PossibleDtypes.supervised, [
         'hyperparams', 'num_process', 'cat_process', 'datetime_process',
         'pretrained_bases', 'mycelia_bases', 'features', 'label', 'split'
     ]), (PossibleDtypes.image, ['hyperparams']),
     (PossibleDtypes.text, ['hyperparams']),
     (PossibleDtypes.fasttext, ['hyperparams']),
     (PossibleDtypes.edit, ['hyperparams'])])
def test_kwargs_possibilities(dtype, keys):
    params = kwargs_possibilities(dtype)
    assert list(params.keys()) == keys


@pytest.mark.parametrize('dtype, body, error', [
    (PossibleDtypes.selfsupervised, {
        'hyperparams': {
            'model_name': 'transformers'
        }
    }, 'wrong_subkey'),
    (PossibleDtypes.image, {
        'cat_process': {
            'fill_value': 0
        }
    }, 'wrong_key'),
    (PossibleDtypes.supervised, {
        'pretrained_bases': [{
            'db_parent': 'test'
        }],
        "label": {
            "label_name": "Label",
            "task": "Regression"
        }
    }, 'missing_must_key'),
])
def test_possible_kwargs_validation(dtype, body, error):
    if error == 'wrong_key':
        with pytest.raises(ValueError) as e:
            kwargs_validation(dtype, **body)
        error_msg = f'Inserted argument `cat_process` is not a valid one for dtype="{dtype}". '\
            f'Please check the documentation and try again.'
        assert e.value.args[0] == error_msg
    elif error == 'wrong_subkey':
        with pytest.raises(ValueError) as e:
            kwargs_validation(dtype, **body)
        error_msg = f'Inserted argument `model_name` is not a valid one for dtpe="{dtype}".'\
            f' Please check the documentation and try again.'
        assert e.value.args[0] == error_msg
    elif error == 'missing_must_key':
        with pytest.raises(ValueError) as e:
            kwargs_validation(dtype, **body)
        assert e.value.args[
            0] == f'[\'id_name\'] parameter is required for the dtype "{dtype}".'


@pytest.mark.parametrize('dtype, body', [
    (PossibleDtypes.supervised, {
        'mycelia_bases': {
            'db_parent': 'test'
        },
        "label": {
            "label_name": "Label",
            "task": "Regression"
        }
    }),
])
def test_wrong_pretrained_bases_input(dtype, body):
    with pytest.raises(DeprecatedError) as e:
        kwargs_validation(dtype, **body)
    assert e.value.args[
        0] == "`mycelia_bases` has been deprecated, please use `pretrained_bases` instead."


def test_check_kwargs_exception():
    with pytest.raises(ValueError) as e:
        kwargs_validation(db_type="Supervised")
    assert e.value.args[
        0] == f"Missing the required arguments: `label`. Please check the documentation and try again."

    with pytest.raises(ValueError) as e:
        kwargs_validation(
            db_type="SelfSupervised",
            **{'trained_bases': {
                'db_parent': 'test',
                'id_name': 'test'
            }})
    assert e.value.args[0] == f'Inserted argument `trained_bases` is not a valid one for dtype="SelfSupervised".'\
                    f' Please check the documentation and try again.'
