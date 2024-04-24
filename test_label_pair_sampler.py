#!/usr/bin/env python3

# MIT License

# Copyright (c) 2024 Derek King

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from label_pair_sampler import LabelPairSampler
from typing import Iterator, List, Tuple


class FakeDataSet:
    def __init__(self, labels: List[int]):
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[Tuple[None, int]]:
        for label in self.labels:
            yield (None, label)


def count_label_pairs(dataset: FakeDataSet, idxs: List[int]) -> int:
    # sampler returns indexes to samples,
    labels = [dataset.labels[idx] for idx in idxs]
    pair_count = 0
    i = 1
    while i < len(labels):
        if labels[i] == labels[i-1]:
            # only count pairs once
            pair_count += 1
            i += 2
        else:
            i += 1
    return pair_count


def validate_sampler(dataset: FakeDataSet,
                     sampler, *,
                     expected_pair_count=None,
                     ):
    expected_batch_size = sampler.batch_size
    if expected_pair_count is None:
        expected_pair_count = len(dataset)//2

    assert len(sampler) == len(dataset)
    idx_set = set()
    count = 0
    small_batch_count = 0
    pair_count = 0
    for idxs in sampler:
        print(idxs)
        count += len(idxs)
        assert 0 < len(idxs) <= expected_batch_size
        if len(idxs) != expected_batch_size:
            small_batch_count += 1
        idx_set.update(idxs)
        pair_count += count_label_pairs(dataset, idxs)
    assert count == len(dataset)
    assert len(idx_set) == len(dataset)
    assert small_batch_count <= 1
    assert pair_count == expected_pair_count


def test_2type_even_full_batch_sampler():
    # There is an even number of samples, and there are a multiple of 4 samples
    # so should get the same number of samples for every batch
    labels = [1]*2 + [2]*6
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=4)
    validate_sampler(dataset, sampler)


def test_2type_even_small_batch_sampler():
    # There is an even number of each label, but number of samples is not the
    # same as the batch size so there will be one small batch
    labels = [1]*4 + [2]*6
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=4)
    validate_sampler(dataset, sampler)


def test_2type_single_odd_sampler():
    # There is an odd number of samples for one label, so one batch will have only 3 labels
    labels = [1]*4 + [2]*3
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=4)
    validate_sampler(dataset, sampler)


def test_2type_all_odd_sampler():
    # both labels have an odd number of samples
    labels = [1]*5 + [2]*3
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=4)
    validate_sampler(dataset, sampler, expected_pair_count=3)


def test_multitype_many_odd_sampler_4batch():
    labels = [1]*5 + [2]*3 + [3]*7 + [4]*10
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=4)
    validate_sampler(dataset, sampler, expected_pair_count=11)


def test_multitype_many_odd_sampler_8batch():
    labels = [1]*5 + [2]*3 + [3]*7 + [4]*10
    dataset = FakeDataSet(labels)
    sampler = LabelPairSampler(dataset, batch_size=8)
    validate_sampler(dataset, sampler, expected_pair_count=11)
