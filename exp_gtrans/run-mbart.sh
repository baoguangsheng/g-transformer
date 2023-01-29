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

data=$1
mode=$2
exp_path=$3

slang=en
tlang=de

echo `date`, data: $data, mode: $mode, exp_path: $exp_path, slang: $slang, tlang: $tlang
bin_path=$exp_path/$data.binarized.$slang-$tlang

run_path=$exp_path/run-mbart
mkdir -p $run_path
cp_path=$run_path/$data.checkpoints.$slang-$tlang
res_path=$run_path/$data.results.$slang-$tlang
echo `date`, run path: $run_path

# ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
mbart_langs=ar,cs,de,en,es,et,fi,fr,gu,hi,it,ja,kk,ko,lt,lv,my,ne,nl,ro,ru,si,tr,vi,zh
mbart_bpe=mbart.cc25/sentence.bpe.model
mbart_model=mbart.cc25/model.pt

if [ $mode == "train" ]; then
  echo `date`, Training model...
  python train.py $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 222 --fp16 --num-workers 4 \
         --task translation_doc --arch mbart_large --source-lang $slang --target-lang $tlang --langs $mbart_langs \
         --doc-mode partial --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
         --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
         --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 \
         --criterion label_smoothed_cross_entropy --label-smoothing 0.2 --dropout 0.3 --attention-dropout 0.1 \
         --max-tokens 2048 --update-freq 2 --validate-interval 1 --patience 5 --no-epoch-checkpoints \
         --restore-file $mbart_model --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
         --load-partial --encoder-ctxlayers 2 --decoder-ctxlayers 2 --cross-ctxlayers 2 \
         --ddp-backend no_c10d > $run_path/train.$data.$slang-$tlang.log 2>&1
elif [ $mode == "test" ]; then
  mkdir -p $res_path
  echo `date`, Testing model on test dataset...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt \
         --gen-subset test --batch-size 32 --beam 5 --max-len-a 1.2 --max-len-b 10 \
         --task translation_doc --source-lang $slang --target-lang $tlang \
         --doc-mode partial --tokenizer moses --bpe 'sentencepiece' --sentencepiece-vocab $mbart_bpe --sacrebleu \
         --langs $mbart_langs --gen-output $res_path/test > $run_path/test.$data.$slang-$tlang.log 2>&1
else
  echo Unknown mode ${mode}.
fi
