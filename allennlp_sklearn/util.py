import collections.abc
from typing import Iterable, Tuple

from allennlp.training.metrics import Metric


def get_metric_name_value_pairs(metric: Metric, default_name: str, reset: bool = False) -> Iterable[Tuple[str, float]]:
    """
    Return the metric as in `Metric.get_metric` but as an iterable of string-float pairs.
    """
    value = metric.get_metric(reset)
    if isinstance(value, collections.abc.Mapping):
        for sub_name, sub_value in value.items():
            if isinstance(sub_value, collections.abc.Iterable):
                for i, sub_value_i in enumerate(sub_value):
                    yield f"{sub_name}_{i}", sub_value_i
            else:
                yield sub_name, sub_value
    elif isinstance(value, collections.abc.Iterable):
        for i, sub_value in enumerate(value):  # type: ignore
            yield f"{default_name}_{i}", sub_value  # type: ignore
    else:
        yield default_name, value
