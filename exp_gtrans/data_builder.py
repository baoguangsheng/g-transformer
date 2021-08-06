#!/usr/bin/env python3 -u
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import os.path as path
import numpy as np
from utils import load_lines_special, save_lines

logger = logging.getLogger()


def convert_to_segment(args):
    # Doc with sequence tag assigned to sentences
    def _segment_seqtag(src, tgt, num=None):
        src = src.split('</s>')
        tgt = tgt.split('</s>')
        segs = []  # [max_tokens, max_segs, src, tgt]
        for idx, (s, t) in enumerate(zip(src, tgt)):
            if len(s) == 0 and len(t) == 0:
                continue
            assert len(s) > 0 and len(t) > 0
            s_toks = s.split()
            t_toks = t.split()
            max_toks = max(len(s_toks), len(t_toks)) + 2
            # count tokens
            if len(segs) > 0 and segs[-1][0] + max_toks < args.max_tokens \
                    and (num is None or segs[-1][1] < num):
                segs[-1][0] += max_toks
                segs[-1][1] += 1
                segs[-1][2] += ['<s> %s </s>' % ' '.join(s_toks)]
                segs[-1][3] += ['<s> %s </s>' % ' '.join(t_toks)]
            else:
                segs.append([max_toks, 1, ['<s> %s </s>' % ' '.join(s_toks)], ['<s> %s </s>' % ' '.join(t_toks)]])
        # output
        srcs = [' '.join(s) for _, _, s, _ in segs]
        tgts = [' '.join(t) for _, _, _, t in segs]
        return srcs, tgts

    logger.info('Building segmented data: %s' % args)
    # specify the segment function
    seg_func = _segment_seqtag

    # train, valid, test
    corpuses = args.corpuses.split(',')
    for corpus in corpuses:
        source_lang_file = '%s.%s' % (corpus, args.source_lang)
        target_lang_file = '%s.%s' % (corpus, args.target_lang)
        src_lines = load_lines_special(path.join(args.datadir, source_lang_file))
        tgt_lines = load_lines_special(path.join(args.datadir, target_lang_file))
        # verify the data
        assert len(src_lines) == len(tgt_lines)
        src_sents = [len(line.split('</s>')) for line in src_lines]
        tgt_sents = [len(line.split('</s>')) for line in tgt_lines]
        assert np.all(np.array(src_sents) == np.array(tgt_sents))
        # convert
        processed = []
        src_data = []
        tgt_data = []
        for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            # check min doc length for training
            if corpus == 'train' and len(src.split()) < args.min_train_doclen:
                logger.warning('Skip too short document: corpus=train, doc=%s, sents=%s, tokens=%s'
                               % (idx, len(src.split('</s>')), len(src.split())))
                continue
            # segment the doc
            srcs, tgts = seg_func(src, tgt, args.max_sents)
            # verify doc length
            srcs_len = [len(line.split()) for line in srcs]
            if any(l > args.max_tokens for l in srcs_len):
                logger.warning('Source doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
                                % (corpus, idx, len(src.split('</s>')), max(srcs_len), args.max_tokens))
            tgts_len = [len(line.split()) for line in tgts]
            if any(l > args.max_tokens for l in tgts_len):
                logger.warning('Target doc has too long segment: corpus=%s, doc=%s, sents=%s, seg_len=%s, max_len=%s.'
                                % (corpus, idx, len(tgt.split('</s>')), max(tgts_len), args.max_tokens))
            # persist
            src_data.extend(srcs)
            tgt_data.extend(tgts)
            processed.append(idx)
        # remove special token
        if args.no_special_tok:
            src_data = [line.replace('<s> ', '').replace(' </s>', '') for line in src_data]
            tgt_data = [line.replace('<s> ', '').replace(' </s>', '') for line in tgt_data]
        # save segmented language files
        logger.info('Processed %s documents of %s with a max_len of %s.' % (len(processed), corpus, args.max_tokens))
        source_lang_file = '%s.%s' % (corpus, args.source_lang)
        target_lang_file = '%s.%s' % (corpus, args.target_lang)
        source_lang_file = path.join(args.destdir, source_lang_file)
        save_lines(source_lang_file, src_data)
        logger.info('Saved %s lines into %s' % (len(src_data), source_lang_file))
        target_lang_file = path.join(args.destdir, target_lang_file)
        save_lines(target_lang_file, tgt_data)
        logger.info('Saved %s lines into %s' % (len(tgt_data), target_lang_file))


''' Generate aligned parallel text
      </s> - separator between sentences
    e.g.
      X: w1 w2 </s> w3 w4 </s>
      Y: w1 w2 w3 </s> w4 w5 </s>
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpuses", default='test,valid,train')
    parser.add_argument("--source-lang", default='en')
    parser.add_argument("--target-lang", default='de')
    parser.add_argument('--datadir', default='exp_test/nc2016.tokenized.en-de/')
    parser.add_argument("--destdir", default='exp_test/nc2016.segmented.en-de/')
    parser.add_argument("--max-sents", default=1, type=int)
    parser.add_argument("--max-tokens", default=512, type=int)
    parser.add_argument("--min-train-doclen", default=-1, type=int)
    parser.add_argument('--no-special-tok', action='store_true', default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='./data_builder.log', format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    os.makedirs(args.destdir, exist_ok=True)
    args.tempdir = path.join(args.destdir, 'tmp')
    os.makedirs(args.tempdir, exist_ok=True)

    convert_to_segment(args)

