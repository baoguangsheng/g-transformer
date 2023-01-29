# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Created by Guangsheng Bao on 11/29/2020

import torch
import math
from fairseq import metrics, utils, search
from fairseq.data import LanguagePairDataset, DocNoiseDataset

from .translation import load_langpair_dataset, TranslationTask
from . import register_task


@register_task('translation_doc')
class DocTranslationTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on
        # Guangsheng Bao: new arguments for doc-NMT
        parser.add_argument('--gen-output', type=str, default='',
                            help='output prefix for ref/gen sentences.')
        parser.add_argument('--doc-mode', default='full', choices=['full', 'partial'],
                            help='work mode as a document-level NMT.'
                                 'full - a normal transformer.' 
                                 'partial - transformer with local/global attention.')
        parser.add_argument('--doc-attn-entropy', default=True, type=bool,
                            help='record attention entropy')

        parser.add_argument('--doc-noise-mask', default=0.0, type=float,
                            help='alias word-dropout, denoting the ratio of tokens to mask out.')
        parser.add_argument('--doc-noise-epochs', default=0, type=int,
                            help='epochs for ramping up noise mask.')
        parser.add_argument('--doc-double-lr', action='store_true',
                            help='double learning rate, one for pretrained, one for randinit.')

        parser.add_argument('--load-partial', action='store_true',
                            help='initialize the model with pretrained parameters.')
        parser.add_argument('--load-partial-global-from-local', action='store_true',
                            help='initialize global attention from the parameters of local attention.')
        parser.add_argument('--lr-scale-pretrained', default=0.2, type=float,
                            help='lr scale for pretrained parameters when fine-tuning.')
        parser.add_argument('--weight-decay-randinit', default=0, type=float,
                            help='weight decay for random-initialized parameters.')

        parser.add_argument('--encoder-ctxlayers', default=2, type=int,
                            help='how many layers for global attention.')
        parser.add_argument('--decoder-ctxlayers', default=2, type=int,
                            help='how many layers for global attention.')
        parser.add_argument('--cross-ctxlayers', default=2, type=int,
                            help='how many layers for global attention.')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        for d in [src_dict, tgt_dict]:
            for l in self.langs:
                d.add_symbol('[{}]'.format(l))
            d.add_symbol('<mask>')
        self.naddition = len(self.langs) + 1

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        attn_keys = ['encoder_self_local', 'encoder_self_global',
                     'decoder_self_local', 'decoder_self_global',
                     'decoder_cross_local', 'decoder_cross_global']

        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        for key in attn_keys:
            val_sum = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(key, val_sum / nsentences / math.log(2), nsentences, round=3)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        dataset = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'prepend_bos', False),
            source_append_langid=False,
            target_append_langid=True,
        )

        if split in ['train']:
            if self.args.doc_noise_mask > 0:
                dataset = DocNoiseDataset(
                    dataset, seed=self.args.seed, naddition=self.naddition,
                    doc_noise_mask=self.args.doc_noise_mask, doc_noise_epochs=self.args.doc_noise_epochs)

        self.datasets[split] = dataset

    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            # Choose search strategy. Defaults to Beam Search.
            sampling = getattr(args, "sampling", False)
            sampling_topk = getattr(args, "sampling_topk", -1)
            sampling_topp = getattr(args, "sampling_topp", -1.0)
            assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
            assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

            if sampling:
                search_strategy = search.Sampling(
                    self.target_dictionary, sampling_topk, sampling_topp
                )
            else:
                search_strategy = search.BeamSearch(self.target_dictionary)

            return SequenceGenerator(
                models,
                self.source_dictionary,
                self.target_dictionary,
                force_length=getattr(args, 'doc_mode', 'full') == 'partial',
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                eos=self.tgt_dict.index('[{}]'.format(self.args.target_lang)),
                search_strategy=search_strategy)

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary)
        return dataset
