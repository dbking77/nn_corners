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

from collections import defaultdict
from torch.utils.data import Sampler
from typing import Iterator, List
import random

class LabelPairSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int=2) -> None:
        if batch_size % 2 != 0:
            raise InvalidArgumentError("batch_size should be even value")
        self.batch_size = batch_size
        self.len = len(dataset)
        self.label_idxs = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_idxs[label].append(idx)

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> Iterator[List[int]]:
        sample_pairs = []
        singles = []
        for label, idxs in self.label_idxs.items():
            idxs = idxs[:]
            random.shuffle(idxs)
            for i in range(0, len(idxs)-1, 2):
                sample_pairs.append( [idxs[i], idxs[i+1] ])
            if len(idxs) & 1:
                singles.append(idxs[-1])
        random.shuffle(sample_pairs)
        random.shuffle(singles)

        batches = []
        half_batch_size = self.batch_size//2
        for i in range(0, len(sample_pairs), half_batch_size):
            batch_idxs = sum(sample_pairs[i:i+half_batch_size], [])
            batches.append(batch_idxs)

        # last batch might not be full, so fill it out with singles
        while len(singles) and (len(batches[-1]) < self.batch_size):
            batches[-1].append(singles.pop())
            
        # divide up all remaining singles into batches
        for i in range(0, len(singles), self.batch_size):
            batch_idxs = singles[i:i+self.batch_size]
            batches.append(batch_idxs)

        random.shuffle(batches)
        for batch in batches:
            yield batch