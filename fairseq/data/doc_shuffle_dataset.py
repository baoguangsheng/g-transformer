# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math
import scipy.stats as sps
import logging
from . import data_utils, FairseqDataset

logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target_prev',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class DocShuffleDataset(FairseqDataset):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            sample = self.dataset[index]
            src = sample['source']
            tgt = sample['target']
            src, tgt = self.shuffle(src, tgt)
            target = tgt
            source = src
            target_prev = tgt

        return {
            'id': index,
            'source': source,
            'target': target,
            'target_prev': target_prev,
        }

    def shuffle(self, src, tgt):
        def _split(doc, dict):
            doc = doc.cpu().numpy().tolist()
            res = [[]]
            for tok in doc:
                res[-1].append(tok)
                if tok == dict.eos_index:
                    res.append([])
            if len(res[-1]) == 0:
                res = res[:-1]
            return res

        src_sents = _split(src, self.dataset.src_dict)
        tgt_sents = _split(tgt, self.dataset.tgt_dict)
        assert len(src_sents) == len(tgt_sents) - 1
        idx = np.arange(len(src_sents))
        np.random.shuffle(idx)
        res_src = sum([src_sents[i] for i in idx], [])
        res_tgt = sum([tgt_sents[i] for i in idx], []) + tgt_sents[-1]
        return torch.tensor(res_src, device=src.device), torch.tensor(res_tgt, device=tgt.device)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return collate(
            samples,
            pad_idx=self.dataset.src_dict.pad(),
            eos_idx=self.dataset.eos,
            left_pad_source=self.dataset.left_pad_source,
            left_pad_target=self.dataset.left_pad_target,
            input_feeding=self.dataset.input_feeding,
        )

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch
