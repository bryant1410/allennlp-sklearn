from typing import Callable, Iterable, Mapping, Optional, Union

import joblib
import numpy as np
import torch
from allennlp.common import Registrable
from numpy.random.mtrand import RandomState
from overrides import overrides
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from typing_extensions import Literal


class SklearnEstimator(nn.Module, Registrable):
    """

    It's a PyTorch Module to have model saving, hooks and `training` (so the attribute is passed from upstream as well).

    `fit_on_test==True` makes sense when you want to overfit, e.g., when you want test a majority baseline
    which changes the class balance between train and test.
    """

    def __init__(self, estimator: BaseEstimator, fit_on_test: bool = False, ignore_predict_proba: bool = False) -> None:
        super().__init__()
        self.estimator = estimator
        self.fit_on_test = fit_on_test
        self.ignore_predict_proba = ignore_predict_proba

        self._already_fit = False

    @overrides
    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Mapping[str, torch.Tensor]:
        # PyTorch tensors are gonna be converted into NumPy arrays by the scikit-learn models.
        # Internally, that implies sending them to the CPU and raising an exception if they require gradients,
        # which is expected. The input tensors shouldn't require gradients.

        if (self.training or self.fit_on_test) and y is not None:
            partial_fit_func = getattr(self.estimator, "partial_fit", None)
            if partial_fit_func:
                partial_fit_func(X, y)
            elif self._already_fit and not (self.fit_on_test and not self.training):
                raise AlreadyFittedError(
                    "The underlying scikit-learn estimator doesn't support online learning "
                    "(it doesn't provide a `partial_fit` method) and it was already fitted."
                )
            else:
                self.estimator.fit(X, y)  # noqa
                self._already_fit = True

        try:
            # The AllenNLP metrics need the scores.
            # We return the probabilities as the scores if they're available within the underlying estimator,
            # otherwise we just return the score 1 for the predicted class and 0 for the rest.
            # We don't check if the method exists, as in `partial_fit`, because it may exist but may not be usable,
            # such as `SVC` when `probability==False`.
            try:
                if self.ignore_predict_proba:
                    raise AttributeError("`ignore_predict_proba` is set to true.")
                else:
                    scores = torch.from_numpy(self.estimator.predict_proba(X))  # noqa
            except AttributeError:
                # If the estimator doesn't define `predict`, it's gonna fail and it's okay that it happens.
                y_pred = self.estimator.predict(X)  # noqa

                # TODO: should add support for regression, multi-target regression and multi-label classification.
                #   It could be done by telling what's len(y.shape), then checking if it's classification or regression.
                #       (`_estimator_type`).
                #   For now, it only supports single-label classification.
                # FIXME: Not sure if `.classes_` is always present in classifiers.
                scores = torch.zeros((y.shape[0], len(self.estimator.classes_)))  # noqa
                scores[torch.arange(y_pred.shape[0]), y_pred] = 1
        except NotFittedError as e:
            raise NotFittedError(
                "The underlying scikit-learn estimator was not fitted."
                "You need to first call this instance with labels and in training mode to fit it."
            ) from e

        return {"scores": scores}


class SklearnTransformerEstimator(SklearnEstimator, TransformerMixin):
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.transform(X)  # noqa


class AlreadyFittedError(ValueError, AttributeError):
    pass


@SklearnEstimator.register("dummy_classifier")
class DummyClassifierEstimator(SklearnEstimator):
    def __init__(self, strategy: Literal["stratified", "most_frequent", "prior", "uniform", "constant"] = "warn",
                 random_state: Optional[Union[int, RandomState]] = None,
                 constant: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
                 _fit_on_test: bool = False, _ignore_predict_proba: bool = False) -> None:
        super().__init__(DummyClassifier(strategy=strategy, random_state=random_state, constant=constant),
                         fit_on_test=_fit_on_test, ignore_predict_proba=_ignore_predict_proba)


@SklearnEstimator.register("svc")
class SvcEstimator(SklearnEstimator):
    def __init__(self, C: float = 1.0,
                 kernel: Union[Callable[[np.ndarray, np.ndarray], np.ndarray],
                               Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]] = "rbf", degree: int = 3,
                 gamma: Union[float, Literal["scale", "auto"]] = "scale", coef0: float = 0.0, shrinking: bool = True,
                 probability: bool = False, tol: float = 1e-3, cache_size: int = 200,
                 class_weight: Optional[Union[Mapping[int, float], Literal["balanced"]]] = None, verbose: bool = False,
                 max_iter: int = -1, decision_function_shape: Literal["ovo", "ovr"] = "ovr", break_ties: bool = False,
                 random_state: Optional[Union[int, RandomState]] = None, _fit_on_test: bool = False,
                 _ignore_predict_proba: bool = False) -> None:
        super().__init__(SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                             probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                             verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                             break_ties=break_ties, random_state=random_state),
                         fit_on_test=_fit_on_test, ignore_predict_proba=_ignore_predict_proba)


@SklearnEstimator.register("pipeline")
class PipelineEstimator(SklearnEstimator):
    def __init__(self, estimators: Iterable[SklearnEstimator], memory: Optional[Union[str, joblib.Memory]] = None,
                 verbose: bool = False, _fit_on_test: bool = False, _ignore_predict_proba: bool = False) -> None:
        super().__init__(make_pipeline(*(e.estimator for e in estimators), memory=memory, verbose=verbose),
                         fit_on_test=_fit_on_test, ignore_predict_proba=_ignore_predict_proba)


@SklearnEstimator.register("standard_scaler")
class StandardScalerTransformerEstimator(SklearnTransformerEstimator):
    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True,
                 _fit_on_test: bool = False, _ignore_predict_proba: bool = False) -> None:
        super().__init__(StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std),
                         fit_on_test=_fit_on_test, ignore_predict_proba=_ignore_predict_proba)

# TODO: add all estimators
