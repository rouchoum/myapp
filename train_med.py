import argparse
from itertools import chain, starmap
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from typing import Iterable, Tuple, Type, Mapping, Any, Sequence, Union, List

from data_loader.loader import load_raw_signals
from data_loader.data_cleaning import clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import (flat_f1_score,
                                      flat_recall_score,
                                      flat_precision_score,
                                      flat_accuracy_score)
from encoding.stats_encoder import StatsEncoder
from encoding.shapelet_encoder import ShapeletEncoder
from encoding.time_splitter import TimeSplitter
from encoding.reduce_target_chunk import TargetChunkReducer
from encoding.crf_encoder import CRFEncoder
from encoding.compose_encoder import ComposeEncoder
from encoding.ohe_timeseries import OheTimeSeries
from preprocessor.scaling import TimeSeriesScaler
from preprocessor.smoothing import TSmoother
from preprocessor.stationarity import StationarityFixer
from models.rnn import get_model
from models.standard_classifier import TimeSeriesClassifier
from sklearn.base import BaseEstimator
from utils.hyperparams import transform_params


class Trainer(BaseEstimator):

    def __init__(self,
                 model_type: str,
                 encoder_type: str,
                 model_params: Mapping[str, Any],
                 stationarity_params: Mapping[str, Any],
                 smooth_params: Mapping[str, Any],
                 scaler_params: Mapping[str, Any],
                 time_splitter_params: Mapping[str, Any],
                 encoder_params: Mapping[str, Any],
                 metric: str = None):

        self.model_type = model_type
        self.encoder_type = encoder_type
        self.model_params = model_params
        self.stationarity_params = stationarity_params
        self.smooth_params = smooth_params
        self.scaler_params = scaler_params
        self.time_splitter_params = time_splitter_params
        self.encoder_params = encoder_params
        self.metric = metric

        self.model = None
        self.label_transformer = None
        self.init_pipelines()

    def init_pipelines(self):
        encoding_steps = [
            ('stationary', StationarityFixer(**self.stationarity_params)),
            ('smoother', TSmoother(**self.smooth_params)),
            ('scaler', TimeSeriesScaler(**self.scaler_params)),
            ('time_splitter', TimeSplitter(**self.time_splitter_params)),
        ]
        label_transformer_params = [
            ('time_splitter', TimeSplitter,
             dict([*self.time_splitter_params.items(), ('tag_status', True)])),
            ('reduce_chunk', TargetChunkReducer, {})
        ]

        if self.encoder_type == 'stats':
            encoder_constructor = StatsEncoder
        else:
            encoder_constructor = ShapeletEncoder

        if self.model_type == 'crf':
            self.label_transformer = Trainer.build_label_transformer(
                label_transformer_params)
            crf = CRF(**self.model_params)
            encoding_steps.append(('encoder',
                                   encoder_constructor(**self.encoder_params)))
            encoding_steps.append(('crf_encoder', CRFEncoder()))
            encoding_steps.append(('model', crf))
            self.model = Pipeline(encoding_steps)

        elif self.model_type == 'rnn':
            rnn = get_model(**self.model_params)
            encoding_steps.append(('encoder',
                                   encoder_constructor(**self.encoder_params)))
            encoding_steps.append(('crf_encoder', CRFEncoder()))
            encoding_steps.append(('model', rnn))
            self.model = Pipeline(encoding_steps)

            label_transformer_params.append(("ohe", OheTimeSeries, {}))
            self.label_transformer = Trainer.build_label_transformer(
                label_transformer_params)

        elif self.model_type in ('SVC', 'RandomForestClassifier'):
            self.label_transformer = Trainer.build_label_transformer(
                label_transformer_params)

            classifier = TimeSeriesClassifier(**self.model_params)
            encoding_steps.append(('encoder',
                                   encoder_constructor(**self.encoder_params)))
            encoding_steps.append(('model', classifier))

            self.model = Pipeline(encoding_steps)

    def fit(self, X, y):
        self.label_transformer.fit(y)
        return self.model.fit(X, self.label_transformer.transform(y))

#### nouvelle_fonction
    def y_fit(self, y):
        self.label_transformer.fit(y)
        return self.label_transformer.transform(y)
    ##fin

    def predict(self, X):
        if self.model_type == 'crf':
            return self.model.predict(X)
        if self.model_type == 'rnn':
            return np.argmax(self.model.predict(X), axis=-1)
        if self.model_type in {'SVC', 'RandomForestClassifier'}:
            return self.model.predict(X, keep_batch_dim=True)
        raise ValueError(f'Unsupported model type {self.model_type}')

    def score(self,
              X, y,
              metric: Union[str, Sequence[str]] = None
              ) -> Union[float, List[float]]:
        if not metric:
            metric = 'f1' if not self.metric else self.metric

        y_transformed = self.label_transformer.transform(y)
        y_pred = self.predict(X)

        if type(metric) == str:
            return Trainer.compute_metric(y_transformed, y_pred, metric=metric)
        else:
            return [Trainer.compute_metric(y_transformed, y_pred, metric=m)
                    for m in metric]

    @staticmethod
    def compute_metric(y_true: Iterable[np.ndarray],
                       y_pred: Iterable[np.ndarray],
                       metric: str,
                       score_average_method: str = 'weighted') -> float:
        if metric == 'f1':
            return flat_f1_score(y_true, y_pred, average=score_average_method)
        if metric == 'precision':
            return flat_precision_score(y_true, y_pred,
                                        average=score_average_method)
        if metric == 'recall':
            return flat_recall_score(y_true, y_pred,
                                     average=score_average_method)
        if metric == 'accuracy':
            return flat_accuracy_score(y_true, y_pred)

        raise ValueError(f'Unsupported metric {metric}')

    def save(self, save_model: str = None, save_label_transformer: str = None):
        if save_model:
            with open(save_model, 'wb') as f:
                pickle.dump(self.model, f)

        if save_label_transformer:
            with open(save_label_transformer, 'wb') as f:
                pickle.dump(self.label_transformer, f)

    @staticmethod
    def build_label_transformer(
            params: Iterable[Tuple[str, Type, Mapping[str, Any]]]):
        transformed_params = starmap(lambda prefix, constructor, param_dict:
                                     transform_params(prefix,
                                                      param_dict).items(),
                                     params)
        all_transformed_params = dict(chain.from_iterable(transformed_params))

        prefixed_constructors = list(map(lambda transform_data:
                                         (transform_data[0], transform_data[1]),
                                         params))

        label_transformer = ComposeEncoder(prefixed_constructors,
                                           all_transformed_params)

        return label_transformer


def cross_validate(model_type: str,
                   encoder_type: str,
                   metric: str,
                   X: Sequence[pd.DataFrame],
                   y: Sequence[pd.DataFrame],
                   model_params: Mapping[str, Any],
                   stationarity_params: Mapping[str, Any],
                   smooth_params: Mapping[str, Any],
                   scaler_params: Mapping[str, Any],
                   time_splitter_params: Mapping[str, Any],
                   encoder_params: Mapping[str, Any],
                   cv: int = 3,
                   model_fname: str = '',
                   label_transform_fname: str = ''
                   ) -> float:

    trainer = Trainer(model_type,
                      encoder_type,
                      model_params,
                      stationarity_params,
                      smooth_params,
                      scaler_params,
                      time_splitter_params,
                      encoder_params,
                      metric)

    scores = cross_val_score(trainer, X, y, cv=cv, n_jobs=min(8, cv))

    trainer.save(model_fname, label_transform_fname)

    return np.mean(scores)


def train(model_type: str,
          encoder_type: str,
          metric: Union[str, Iterable[str]],
          X: Sequence[pd.DataFrame],
          y: Sequence[pd.DataFrame],
          model_params: Mapping[str, Any],
          stationarity_params: Mapping[str, Any],
          smooth_params: Mapping[str, Any],
          scaler_params: Mapping[str, Any],
          time_splitter_params: Mapping[str, Any],
          encoder_params: Mapping[str, Any],
          split_factor: float = .8,
          model_fname: str = '',
          label_transform_fname: str = '',
          val_samples_fname: str = '',
          samples_ids: Sequence[Union[str, int]] = None) -> float:

    if val_samples_fname and not samples_ids:
        raise ValueError('samples_ids must be provided to save the val samples '
                         f'indices in {val_samples_fname}')

    x_train, x_test, y_train, y_test = train_test_split(X, y,train_size=split_factor,random_state=42)

    trainer = Trainer(model_type,
                      encoder_type,
                      model_params,
                      stationarity_params,
                      smooth_params,
                      scaler_params,
                      time_splitter_params,
                      encoder_params,
                      metric)

    #new_y=trainer.y_fit( y_train)
    trainer.fit(x_train, y_train)
    score = trainer.score(x_test, y_test)
    #
    # trainer.save(model_fname, label_transform_fname)
    #
    # if val_samples_fname:
    #     with open(val_samples_fname, 'w') as f:
    #         f.write('\n'.join([str(idx) for idx in test_idx]))

    return score

file_dir='/home/mohamed/Downloads/code_mohamed/samples_corrected'
dfs = load_raw_signals(file_dir)
clean_dicts = clean_dataset(dfs, drop_erroneous=True, sampling_period_ms=100)
clean_dfs = list(map(lambda x: x[1], clean_dicts.items()))

feats = clean_dfs
labels = list(map(lambda x: x[["tag"]], clean_dfs))


#training
X=feats
y=labels
metric = 'f1'

score=train('SVC',
           'stats',
            metric,
            feats,
            labels,
            {'classifier_type': 'SVC','classifier_params': {'C': 0.01, 'kernel': 'linear'}},
            {'method': 'align_stable', 'method_params': {'metric': 'mean'}},
            {'method': 'exp', 'smoother_params':{'window_len':2}},
            {'method': 'standard','scaler_params': {}},
            {'chunk_size': 100,'split_by_label': True},
            {'mean':  True, 'var': True},
            0.9)

print(score)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("data")
#     parser.add_argument("--model",
#                         choices=["crf", "rnn", "SVC", "RandomForestClassifier"],
#                         default="crf")
#     parser.add_argument("--encoder",
#                         choices=["shapelet", "stats"],
#                         default="stats")
#     parser.add_argument("--hyper-parameters")
#     parser.add_argument("--model-fname", default="model.pkl")
#     parser.add_argument("--label-transform-fname", default="label_transform.pkl")
#     parser.add_argument("--stationarity-parameters")
#     parser.add_argument("--smoother-parameters")
#     parser.add_argument("--scaler-parameters")
#     parser.add_argument("--time-splitter-parameters")
#     parser.add_argument("--encoder-parameters")
#     parser.add_argument("--val-samples-fname", default="")
#
#     args = parser.parse_args()
#
#     dfs = load_raw_signals(args.data)
#     clean_dicts = clean_dataset(dfs, drop_erroneous=True, sampling_period_ms=100)
#     clean_dfs = list(map(lambda x: x[1], clean_dicts.items()))
#
#     feats = clean_dfs
#     labels = list(map(lambda x: x[['tag']], clean_dfs))
#
#     with open(args.hyper_parameters) as f:
#         model_params = json.load(f)
#     with open(args.stationarity_parameters) as f:
#         stat_params = json.load(f)
#     with open(args.smoother_parameters) as f:
#         smooth_params = json.load(f)
#     with open(args.scaler_parameters) as f:
#         scaler_params = json.load(f)
#     with open(args.time_splitter_parameters) as f:
#         time_splitter_params = json.load(f)
#     with open(args.encoder_parameters) as f:
#         encoder_params = json.load(f)
#
#     split_factor = .9
#     metric = ['f1', 'precision', 'recall', 'accuracy']
#
#     f1, precision, recall, accuracy = train(args.model,
#                                             args.encoder,
#                                             metric,
#                                             feats,
#                                             labels,
#                                             model_params,
#                                             stat_params,
#                                             smooth_params,
#                                             scaler_params,
#                                             time_splitter_params,
#                                             encoder_params,
#                                             split_factor,
#                                             args.model_fname,
#                                             args.label_transform_fname,
#                                             args.val_samples_fname,
#                                             list(clean_dicts.keys()))
#     print(f'Precision: {precision}')
#     print(f'Accuracy: {accuracy}')
#     print(f'Recall: {recall}')
#     print(f'F1 score: {f1}')
