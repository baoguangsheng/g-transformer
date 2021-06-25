#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
# e.g.
# bash run-sent.sh data exp_test iwslt17
# bash run-sent.sh train exp_test iwslt17
# bash run-sent.sh test exp_test iwslt17

mode=$1
exp_path=$2
data=$3
slang=en
tlang=de
echo mode: $mode, exp_path: $exp_path, data: $data, slang: $slang, tlang: $tlang

tok_path=$exp_path/$data.tokenized.$slang-$tlang
bin_path=$exp_path/$data.binarized.$slang-$tlang
cp_path=$exp_path/$data.checkpoints.$slang-$tlang

if [ $mode == "data" ]; then
  echo Prepraring data...
  bash prepare-sent.sh raw_data/$data $tok_path
  python -m fairseq_cli.preprocess --task translation --joined-dictionary --source-lang $slang --target-lang $tlang \
         --trainpref $tok_path/train --validpref $tok_path/valid --testpref $tok_path/test --destdir $bin_path --workers 8
elif [ $mode == "train" ]; then
  echo Training model...
  python train.py  $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 444 --fp16 --num-workers 4 \
         --task translation --arch transformer_base --source-lang $slang --target-lang $tlang  \
         --share-all-embeddings  --optimizer adam --adam-betas "(0.9, 0.98)" --lr 5e-4 --lr-scheduler inverse_sqrt \
         --warmup-updates 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --no-epoch-checkpoints \
         --max-tokens 4096 --update-freq 1 --validate-interval 1 --patience 10 > $exp_path/train.$data.$slang-$tlang.log 2>&1
elif [ $mode == "test" ]; then
  echo Testing model...
  python -m fairseq_cli.generate $bin_path --path $cp_path/checkpoint_best.pt --batch-size 64 --beam 5 \
         --task translation --source-lang $slang --target-lang $tlang \
         --max-len-a 1.2 --max-len-b 10 --remove-bpe --tokenizer moses --sacrebleu \
         --output $exp_path/$data.$slang-$tlang > $exp_path/test.$data.$slang-$tlang.log 2>&1
else
  echo Unknown mode ${mode}.
fi
