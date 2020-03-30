import sys

from allennlp.data.samplers import BasicBatchSampler, BatchSampler, Sampler


@BatchSampler.register("sklearn")
class SklearnBatchSampler(BasicBatchSampler):
    """A `BasicBatchSampler` which has a very large default batch size."""

    def __init__(self, sampler: Sampler, batch_size: int = sys.maxsize, drop_last: bool = False) -> None:
        super().__init__(sampler, batch_size, drop_last)
