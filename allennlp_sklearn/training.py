import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import torch
from allennlp.common import Tqdm
from allennlp.models import Model
from allennlp.training import Trainer
from allennlp.training.checkpointer import Checkpointer
from overrides import overrides
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@Trainer.register("sklearn")
class SklearnTrainer(Trainer):
    def __init__(self, serialization_dir: str, model: Model, data_loader: DataLoader,
                 validation_data_loader: Optional[DataLoader] = None,
                 checkpointer: Optional[Checkpointer] = None) -> None:
        super().__init__(serialization_dir)
        self.model = model
        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader
        self.checkpointer = checkpointer or Checkpointer(self._serialization_dir)

    @overrides
    def train(self) -> Dict[str, Any]:
        self.model.train()
        logger.info("Training…")
        metrics = {f"training_{k}": v for k, v in self._compute_metrics(self.data_loader)}

        logger.info("Archiving…")
        self.checkpointer.save_checkpoint(epoch="", trainer=self, is_best_so_far=True)

        if self.validation_data_loader:
            self.model.eval()
            logger.info("Validating…")
            for k, v in self._compute_metrics(self.validation_data_loader):
                metrics[f"validation_{k}"] = v

        return metrics

    def _compute_metrics(self, data_loader: DataLoader) -> Iterable[Tuple[str, Any]]:
        with torch.no_grad():
            # We use batches because the dataset may not fit in memory (the iterable was set up thus to batch)
            # and the model estimator may support `partial_fit`. If the estimator doesn't support it,
            # it's gonna raise an exception that the user can see.
            for tensor_dict_batch in Tqdm.tqdm(data_loader):
                self.model(**tensor_dict_batch)
            return self.model.get_metrics(reset=True).items()  # noqa

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        yield self.model.state_dict(), {}
