#!/usr/bin/env python3 -u
# Guangsheng Bao: changed on 2020/10/3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys
import numpy as np
import torch

from fairseq import bleu, checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders
from utils import remove_seps, save_lines
from fairseq.criterions.cross_entropy import CrossEntropyCriterion
from fairseq.data import data_utils

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.dataset_impl == 'raw', \
        '--replace-unk requires a raw text dataset (--dataset-impl=raw)'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)

def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    arg_overrides = eval(args.model_overrides)
    # arg_overrides['doc_mode'] = args.doc_mode
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=arg_overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=not getattr(args, "load_partial", False),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    def _new_scorer(args):
        if args.sacrebleu:
            scorer = bleu.SacrebleuScorer()
        else:
            scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        return scorer

    num_bucket = 5
    def _bucket_name(index):
        names = ['%s-th sent' % i for i in range(num_bucket)]
        return names[index]

    # Test results
    res_sents = []
    res_docs = []
    res_segs = []

    # Generate and compute BLEU score
    scorer = _new_scorer(args)
    scorers = [_new_scorer(args) for i in range(num_bucket)]  # 10 buckets
    scorer_doc = _new_scorer(args)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore={
                            generator.eos,
                        }
                    )
                    # Guangsheng Bao: text output with <s> and </s>
                    seg_target_str = '<s> ' + tgt_dict.string(target_tokens, extra_symbols_to_ignore={generator.eos})

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not args.quiet:
                if src_dict is not None:
                    print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore={
                        generator.eos,
                    }
                )
                # Guangsheng Bao: text output with <s> and </s>
                seg_hypo_str = '<s> ' + tgt_dict.string(hypo['tokens'].int().cpu(), extra_symbols_to_ignore={generator.eos})

                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print('H-{}\t{}\t{}'.format(sample_id, score, hypo_str), file=output_file)
                    # detokenized hypothesis
                    print('D-{}\t{}\t{}'.format(sample_id, score, detok_hypo_str), file=output_file)
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))
                    ), file=output_file)

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(['{}-{}'.format(src_idx, tgt_idx) for src_idx, tgt_idx in alignment])
                        ), file=output_file)

                    if args.print_step:
                        print('I-{}\t{}'.format(sample_id, hypo['steps']), file=output_file)

                    if getattr(args, 'retain_iter_history', False):
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print('E-{}_{}\t{}'.format(sample_id, step, h_str), file=output_file)

                # Score nbest hypothesis
                if has_target:
                    target_str_noseg = remove_seps(target_str)
                    detok_hypo_str_noseg = remove_seps(detok_hypo_str)
                    # Save test results
                    if args.doc_mode == 'partial':
                        assert len(target_str_noseg) == len(detok_hypo_str_noseg)
                        for idx, (t, h) in enumerate(zip(target_str_noseg, detok_hypo_str_noseg)):
                            res_sents.append((sample_id, idx, t, h))
                    target_str_noseg_doc = ' '.join(target_str_noseg)
                    detok_hypo_str_noseg_doc = ' '.join(detok_hypo_str_noseg)
                    res_docs.append((sample_id, target_str_noseg_doc, detok_hypo_str_noseg_doc))
                    res_segs.append((sample_id, seg_target_str, seg_hypo_str))

                    if hasattr(scorer, 'add_string'):
                        if args.doc_mode == 'partial':
                            for t, h in zip(target_str_noseg, detok_hypo_str_noseg):
                                scorer.add_string(t, h)
                        if len(target_str_noseg) != len(detok_hypo_str_noseg):
                            logger.warning('Number of sentences is not matched: %s for target and %s for hypo.'
                                           % (len(target_str_noseg), len(detok_hypo_str_noseg)))
                        num_sents = min(num_bucket, len(target_str_noseg), len(detok_hypo_str_noseg))
                        for idx in range(num_sents):
                            scorers[idx].add_string(target_str_noseg[idx], detok_hypo_str_noseg[idx])
                        scorer_doc.add_string(target_str_noseg_doc, detok_hypo_str_noseg_doc)
                    else:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens_noseg = [tgt_dict.encode_line(sent, add_if_not_exist=True) for sent in
                                                   target_str_noseg]
                            hypo_tokens_noseg = [tgt_dict.encode_line(sent, add_if_not_exist=True) for sent in
                                                 detok_hypo_str_noseg]
                            target_str_noseg_doc = tgt_dict.encode_line(target_str_noseg_doc, add_if_not_exist=True)
                            detok_hypo_str_noseg_doc = tgt_dict.encode_line(detok_hypo_str_noseg_doc, add_if_not_exist=True)
                        if args.doc_mode == 'partial':
                            for t, h in zip(target_tokens_noseg, hypo_tokens_noseg):
                                scorer.add(t, h)
                        if len(target_tokens_noseg) != len(hypo_tokens_noseg):
                            logger.warning('Number of sentences is not matched: %s for target and %s for hypo.'
                                           % (len(target_tokens_noseg), len(hypo_tokens_noseg)))
                        num_sents = min(num_bucket, len(target_tokens_noseg), len(hypo_tokens_noseg))
                        for idx in range(num_sents):
                            scorers[idx].add_string(target_tokens_noseg[idx], hypo_tokens_noseg[idx])
                        scorer_doc.add_string(target_str_noseg_doc, detok_hypo_str_noseg_doc)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample['nsentences']

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning("BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization")
            else:
                logger.warning("If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization")

        if args.doc_mode == 'partial':
            logger.info('[sentence-level] Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
            logger.info('[document-level] Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer_doc.result_string()))
        else:
            logger.info('Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer_doc.result_string()))

        for i in range(num_bucket):
            logger.info(
                'Bucket {}: {}'.format(_bucket_name(i), scorers[i].result_string() if scorers[i].samples > 0 else 'empty.'))

    if len(args.gen_output) > 0:
        if args.doc_mode == 'partial':
            res_sents = sorted(res_sents, key=lambda x: x[0] * 10000 + x[1])
            save_lines(args.gen_output + '.sent.ref', [t for _, _, t, h in res_sents])
            save_lines(args.gen_output + '.sent.gen', [h for _, _, t, h in res_sents])
        res_docs = sorted(res_docs, key=lambda x: x[0])
        save_lines(args.gen_output + '.doc.ref', [t for _, t, h in res_docs])
        save_lines(args.gen_output + '.doc.gen', [h for _, t, h in res_docs])
        res_segs = sorted(res_segs, key=lambda x: x[0])
        save_lines(args.gen_output + '.seg.ref', [t for _, t, h in res_segs])
        save_lines(args.gen_output + '.seg.gen', [h for _, t, h in res_segs])
    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
