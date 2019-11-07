#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import join
import numpy as np
import datasets.base as base
from sts.augmentation import pad_sent_pair, SemEvalAugmentationStrategy, TripletGenerator
from sts import utils as sts
from datasets.base import TextFloatLabelPartition


def split_and_pad_pairs(asents, bsents):
    result = []
    for a, b in zip(asents, bsents):
        apad, bpad = pad_sent_pair(a.split(' '), b.split(' '))
        result.append((apad, bpad))
    return result


def load_partition(path: str, label2int: dict, partition: str):
    with open(join(path, partition, 's1')) as afile, \
            open(join(path, partition, 's2')) as bfile, \
            open(join(path, partition, 'labels')) as simfile:
        a = [f"<s> {line.strip()} </s>" for line in afile.readlines()]
        b = [f"<s> {line.strip()} </s>" for line in bfile.readlines()]
        sim = [label2int[line.strip()] for line in simfile.readlines()]
        return a, b, sim


class SNLI(base.SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str, batch_size: int,
                 augmentation: SemEvalAugmentationStrategy, label2int: dict, batches_per_epoch: int = None):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.augmentation = augmentation
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        atrain, btrain, simtrain = load_partition(path, label2int, 'train')
        adev, bdev, simdev = load_partition(path, label2int, 'dev')
        atest, btest, simtest = load_partition(path, label2int, 'test')
        self.train_sents = self.augmentation.augment(atrain, btrain, simtrain)
        self.dev_sents = np.array(list(zip(split_and_pad_pairs(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(split_and_pad_pairs(atest, btest), simtest)))
        print(f"Train Pairs (with neutrals): {len(atrain)}")
        print(f"Unique Train Sentences (with neutrals): {len(set(atrain + btrain))}")
        print(f"Unique Dev Sentences: {len(set(adev + bdev))}")
        print(f"Unique Test Sentences: {len(set(atest + btest))}")

    @property
    def nclass(self):
        return self.augmentation.nclass()

    # Using float labels because only contrastive and triplet loss are supported

    def training_partition(self) -> base.SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return TextFloatLabelPartition(self.train_sents, self.batch_size, train=True,
                                       batches_per_epoch=self.batches_per_epoch)

    def dev_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.dev_sents, self.batch_size, train=False)

    def test_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.test_sents, self.batch_size, train=False)


class SNLITriplets(base.SimDataset):

    def __init__(self, path: str, vector_path: str, vocab_path: str, batch_size: int, label2int: dict,
                 batches_per_epoch: int = None):
        self.path = path
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.triplet_generator = TripletGenerator('', None)
        self.vocab, n_inv, n_oov = sts.vectorized_vocabulary(vocab_path, vector_path)
        print(f"Created vocabulary with {n_inv} INV and {n_oov} OOV ({int(100 * n_inv / (n_inv + n_oov))}% coverage)")
        triplets, unused_y = self._load_triplet_partition('train')
        adev, bdev, simdev = load_partition(path, label2int, 'dev')
        atest, btest, simtest = load_partition(path, label2int, 'test')
        self.train_sents = np.array(list(zip(triplets, unused_y)))
        self.dev_sents = np.array(list(zip(split_and_pad_pairs(adev, bdev), simdev)))
        self.test_sents = np.array(list(zip(split_and_pad_pairs(atest, btest), simtest)))
        print(f"Unique Dev Sentences: {len(set(adev + bdev))}")
        print(f"Unique Test Sentences: {len(set(atest + btest))}")

    def _load_triplet_partition(self, partition):
        with open(join(self.path, partition, 'anchors')) as afile, \
                open(join(self.path, partition, 'positives')) as posfile, \
                open(join(self.path, partition, 'negatives')) as negfile:
            anchors = [line.strip() for line in afile.readlines()]
            negatives = [line.strip() for line in posfile.readlines()]
            positives = [line.strip() for line in negfile.readlines()]
            triplets = self.triplet_generator.split_and_pad(anchors, positives, negatives)
            unused_y = np.zeros(len(anchors))
            print(f"Triplets recovered: {len(anchors)}")
            print(f"Unique Train Sentences: {len(set(anchors + positives + negatives))}")
            return triplets, unused_y

    @property
    def nclass(self):
        return 0

    # Using float labels because only triplet loss is supported here

    def training_partition(self) -> base.SimDatasetPartition:
        np.random.shuffle(self.train_sents)
        return TextFloatLabelPartition(self.train_sents, self.batch_size, train=True,
                                       batches_per_epoch=self.batches_per_epoch)

    def dev_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.dev_sents, self.batch_size, train=False)

    def test_partition(self) -> base.SimDatasetPartition:
        return TextFloatLabelPartition(self.test_sents, self.batch_size, train=False)