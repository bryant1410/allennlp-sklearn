from typing import Mapping, Optional

import torch
from allennlp.models import Model
from allennlp.training.metrics import Metric, CategoricalAccuracy
from overrides import overrides

from allennlp_sklearn.estimators import SklearnEstimator
from allennlp_sklearn.util import get_metric_name_value_pairs


@Model.register("sklearn")
class SklearnModel(Model):
    """Generic one. It's supposed to be used with an SklearnTrainer."""
    def __init__(self, estimator: SklearnEstimator, metrics: Optional[Mapping[str, Metric]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.estimator = estimator
        self.metrics = metrics or {"accuracy": CategoricalAccuracy()}

    @overrides
    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Mapping[str, torch.Tensor]:
        output_dict = self.estimator(X, y)

        if y is not None:
            for metric in self.metrics.values():
                metric(output_dict["scores"], y)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Mapping[str, float]:
        return {sub_name: value
                for default_name, metric in self.metrics.items()
                for sub_name, value in get_metric_name_value_pairs(metric, default_name, reset)}


# TODO: does it work well if using namespace labels?
# TODO: make it inherit `make_output_human_readable`
