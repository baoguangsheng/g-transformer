#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 iwslt17 train exp_test"
    exit
fi

# run command
data=$1
mode=$2
exp_path=$3

slang=en
tlang=de

echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang
bin_path_sent=$exp_path/$data-sent.binarized.$slang-$tlang
bin_path_doc=$exp_path/$data-doc.binarized.$slang-$tlang

run_path=$exp_path/run-finetune
mkdir -p $run_path
echo `date`, run path: $run_path

cp_path_sent=$run_path/$data-sent.checkpoints.$slang-$tlang
cp_path_doc=$run_path/$data-doc.checkpoints.$slang-$tlang
res_path=$run_path/$data.results.$slang-$tlang
doc_langs=$slang,$tlang

if [ $mode == "train" ]; then
  echo `date`, Training sentence-level model...
  doc_langs=$slang,$tlang
  python train.py $bin_path_sent --save-dir $cp_path_sent --tensorboard-logdir $cp_path_sent --seed 444 --fp16 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode full --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 \
         > $run_path/train.$data-sent.$slang-$tlang.log 2>&1

  echo `date`, Training document-level model...
  sent_model=$cp_path_sent/checkpoint_best.pt
  echo Load sentence model from $sent_model
  echo `date`, Training model...
  python train.py $bin_path_doc --save-dir $cp_path_doc --tensorboard-logdir $cp_path_doc --seed 444 --num-workers 4 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --arch gtransformer_base --doc-mode partial --share-all-embeddings \
         --optimizer adam --adam-betas "(0.9, 0.98)" \
         --lr-scheduler inverse_sqrt --lr 5e-04 --warmup-updates 4000 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 \
         --restore-file $sent_model --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
         --load-partial --doc-double-lr --lr-scale-pretrained 0.2 \
         --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --doc-noise-mask 0.1 --doc-noise-epochs 40 > $run_path/train.$data-doc.$slang-$tlang.log 2>&1
elif [ $mode == "test" ]; then
  mkdir -p $res_path
  echo `date`, Testing model on test dataset...
  python -m fairseq_cli.generate $bin_path_doc --path $cp_path_doc/checkpoint_best.pt \
         --gen-subset test --batch-size 16 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang --langs $doc_langs \
         --doc-mode partial --tokenizer moses --remove-bpe --sacrebleu \
         --gen-output $res_path/test > $run_path/test.$data.$slang-$tlang.log 2>&1
else
  echo Unknown mode ${mode}.
fi