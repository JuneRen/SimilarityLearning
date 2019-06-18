import torch
import numpy as np
from distances import Distance
from metrics import Metric, SpeakerValidationConfig
from datasets.base import SimDatasetPartition
from losses.base import TrainingListener
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.database import get_protocol, get_unique_identifier
from pyannote.metrics.binary_classification import det_curve
from pyannote.core.utils.distance import cdist
from pyannote.core import Timeline
from common import DEVICE


class Evaluator:

    @property
    def metric_name(self):
        raise NotImplementedError

    @property
    def results(self):
        raise NotImplementedError

    def eval(self, model, train_embs, train_y):
        raise NotImplementedError


class EvaluatorTrainingListener(TrainingListener):

    def __init__(self, evaluator: Evaluator, interval: int, keep_train_embeddings: bool = False, callbacks: list = None):
        self.evaluator = evaluator
        self.interval = interval
        self.keep_train_embeddings = keep_train_embeddings
        self.callbacks = callbacks if callbacks is not None else []
        self.best_metric, self.best_epoch = 0, None
        self.train_embs, self.train_y = [], []

    def on_before_epoch(self, epoch):
        self.train_embs, self.train_y = [], []

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        if self.keep_train_embeddings:
            self.train_embs.append(feat.detach().cpu().numpy())
            self.train_y.append(y.detach().cpu().numpy())

    def on_before_train(self, checkpoint):
        if checkpoint is not None:
            self.best_metric = checkpoint['accuracy']

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        if epoch % self.interval == 0:
            metric_value = self.evaluator.eval(model.to_prediction_model(), self.train_embs, self.train_y)
            header = f"--------------- Epoch {epoch:02d} Results ---------------"
            print(header)
            print(f"{self.evaluator.metric_name}: {metric_value:.6f}")
            if self.best_epoch is not None:
                print(f"Best until now: {self.best_metric:.6f}, at epoch {self.best_epoch}")
            print('-' * len(header))
            if metric_value > self.best_metric:
                print(f"New Best {self.evaluator.metric_name}!")
                self.best_metric = metric_value
                self.best_epoch = epoch
                for cb in self.callbacks:
                    test_embs, test_y = self.evaluator.results
                    cb.on_best_accuracy(epoch, model, loss_fn, optim, metric_value, test_embs, test_y)


class SingleForwardMetricEvaluator(Evaluator):

    def __init__(self, partition: SimDatasetPartition, metric: Metric, callbacks: list = None):
        super(SingleForwardMetricEvaluator, self).__init__()
        self.partition = partition
        self.metric = metric
        self.callbacks = callbacks if callbacks is not None else []
        self.last_feat, self.last_y = None, None

    @property
    def metric_name(self):
        return str(self.metric)

    @property
    def results(self):
        return self.last_feat, self.last_y

    def eval(self, model, train_embs, train_y):
        model.eval()

        self.metric.fit(train_embs, train_y)

        for cb in self.callbacks:
            cb.on_before_test()

        feat_test, y_test = [], []
        with torch.no_grad():
            for i in range(self.partition.nbatches()):
                x, y = next(self.partition)
                if isinstance(x, torch.Tensor):
                    x = x.to(DEVICE)
                if isinstance(y, torch.Tensor):
                    y = y.to(DEVICE)

                # Feed Forward
                feat, logits = model(x, y)

                # Track accuracy
                feat = feat.detach().cpu().numpy()
                logits = logits.detach().cpu().numpy() if logits is not None else np.array([])
                y = y.detach().cpu().numpy()
                feat_test.append(feat)
                y_test.append(y)
                self.metric.calculate_batch(feat, logits, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        for cb in self.callbacks:
            cb.on_after_test()

        self.last_feat, self.last_y = np.concatenate(feat_test), np.concatenate(y_test)

        return self.metric.get()


class PairForwardMetricEvaluator(Evaluator):

    def __init__(self, partition: SimDatasetPartition, metric: Metric, callbacks: list = None):
        super(PairForwardMetricEvaluator, self).__init__()
        self.partition = partition
        self.metric = metric
        self.callbacks = callbacks if callbacks is not None else []
        self.last_feat, self.last_y = None, None

    @property
    def results(self):
        return self.last_feat, self.last_y

    @property
    def metric_name(self):
        return str(self.metric)

    def eval(self, model, train_embs, train_y):
        model.eval()

        self.metric.fit(train_embs, train_y)

        for cb in self.callbacks:
            cb.on_before_test()

        feat_test, y_test = [], []
        with torch.no_grad():
            for i in range(self.partition.nbatches()):
                x, y = next(self.partition)
                if isinstance(x, torch.Tensor):
                    x = x.to(DEVICE)
                if isinstance(y, torch.Tensor):
                    y = y.to(DEVICE)

                # Feed Forward
                feat, _ = model(x, y)

                # Track accuracy (we receive a pair of embeddings)
                feat1 = feat[0].detach().cpu().numpy()
                feat2 = feat[1].detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                feat = (feat1, feat2)
                feat_test.append(feat)
                y_test.append(y)
                self.metric.calculate_batch(feat, None, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        for cb in self.callbacks:
            cb.on_after_test()

        self.last_feat = (np.concatenate([lfeat for lfeat, _ in feat_test]), np.concatenate([rfeat for _, rfeat in feat_test]))
        self.last_y = np.concatenate(y_test)

        return self.metric.get()


class SpeakerVerificationTrialEvaluator(Evaluator):

    @staticmethod
    def get_hash(trial_file):
        uri = get_unique_identifier(trial_file)
        try_with = trial_file['try_with']
        if isinstance(try_with, Timeline):
            segments = tuple(try_with)
        else:
            segments = (try_with,)
        return hash((uri, segments))

    def __init__(self, partition: str, batch_size: int, distance: Distance, config: SpeakerValidationConfig):
        super(SpeakerVerificationTrialEvaluator, self).__init__()
        self.partition = partition
        self.batch_size = batch_size
        self.distance = distance
        self.config = config

    @property
    def results(self):
        return None, None

    def eval(self, sim_model, train_embs, train_y):
        model = sim_model.to_prediction_model()
        model.eval()
        sequence_embedding = SequenceEmbedding(model=model,
                                               feature_extraction=self.config.feature_extraction,
                                               duration=self.config.duration,
                                               step=.5 * self.config.duration,
                                               batch_size=self.batch_size,
                                               device=DEVICE)
        protocol = get_protocol(self.config.protocol_name, progress=False, preprocessors=self.config.preprocessors)

        y_true, y_pred, cache = [], [], {}

        for trial in getattr(protocol, f"{self.partition}_trial")():

            # compute embedding for file1
            file1 = trial['file1']
            hash1 = self.get_hash(file1)
            if hash1 in cache:
                emb1 = cache[hash1]
            else:
                emb1 = sequence_embedding.crop(file1, file1['try_with'])
                emb1 = np.mean(np.stack(emb1), axis=0, keepdims=True)
                cache[hash1] = emb1

            # compute embedding for file2
            file2 = trial['file2']
            hash2 = self.get_hash(file2)
            if hash2 in cache:
                emb2 = cache[hash2]
            else:
                emb2 = sequence_embedding.crop(file2, file2['try_with'])
                emb2 = np.mean(np.stack(emb2), axis=0, keepdims=True)
                cache[hash2] = emb2

            # compare embeddings
            dist = cdist(emb1, emb2, metric=self.distance.to_sklearn_metric())[0, 0]
            y_pred.append(dist)
            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred), distances=True)
        # Returning 1-eer because the evaluator keeps track of the highest metric value
        return 1 - eer

    @property
    def metric_name(self):
        return 'EER'
