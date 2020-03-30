# allennlp-sklearn

Use scikit-learn models with AllenNLP.

## Components

* `SklearnEstimator`: a base class for scikit-learn estimators to be used with AllenNLP.
There are a few subclasses for now: `DummyClassifierEstimator`, `SvcEstimator`, `PipelineEstimator`,
and `StandardScalerTransformerEstimator`; corresponding to `DummyClassifier`, `SVC`, `Pipeline`, and `StandardScaler`
from scikit-learn. Feel free to add more as it's likely just little code.
* `SklearnTrainer`: train a model that uses a `SklearnEstimator`.

Optional:

* `SklearnModel`: a simple class that contains a `SklearnEstimator`. You can choose to implement your own if it doesn't
satisfy your needs.
* `SklearnBatchSampler`: a convenient batch sampler that grabs all items from the dataset together, so there's only one
big batch.
