#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
# e.g.
# bash prepare-spm.sh data/iwslt17 exp_test/iwslt17.tokenized.en-de

if [ -d "sentencepiece" ]; then
    echo "sentencepiece already exists, skipping download"
else
    echo 'Cloning sentencepiece repository (for BPE pre-processing)...'
    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece
    mkdir build
    cd build
    cmake ..
    make -j 8
    cd ../..
fi

SPM=sentencepiece/build/src/spm_encode
SPMMODEL=mbart.cc25/sentence.bpe.model

if [ ! -f "$SPM" ]; then
    echo "Please set SPM variable correctly to point to spm_encode."
    exit
fi


data=$1
prep=$2
src=en
tgt=de
lang=$src-$tgt
tmp=$prep/tmp

TOK_DATA=$prep/train.$tgt
if [ -f "$TOK_DATA" ]; then
    echo "Tokenized data $TOK_DATA is already exist, skipping data preparation."
    exit
fi

mkdir -p $tmp $prep

echo "filter out empty lines from original data and split doc with empty line..."
for D in train dev test; do
    sf=$data/concatenated_${src}2${tgt}_${D}_${src}.txt
    tf=$data/concatenated_${src}2${tgt}_${D}_${tgt}.txt
    RD=$D
    if [ $D == "dev" ]; then
        RD=valid
    fi

    rf=$tmp/$RD.$lang.tag
    echo $rf

    paste -d"\t" $sf $tf | \
    grep -v -P "^\s*\t" | \
    grep -v -P "\t\s*$" | \
    sed -e 's/\r//g' > $rf

    cut -f 1 $rf | \
    sed -e 's/^<d>\s*$//g' > $tmp/$RD.$src

    cut -f 2 $rf | \
    sed -e 's/^<d>\s*$//g' > $tmp/$RD.$tgt
done

echo "apply sentencepiece..."
for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        echo "${SPM} ${F} with ${SPMMODEL}..."
        ${SPM} --model=${SPMMODEL} < $tmp/$F > $tmp/$F.spm
    done
done

echo "apply doc-level special tags..."
for L in $src $tgt; do
    for F in train.$L valid.$L test.$L; do
        cat $tmp/$F.spm | \
        # replace empty line with [DOC]
        sed -e 's/^$/[DOC]/g' | \
        # connect all lines into one line
        sed -z -e 's/\n/ [SEP] /g' | \
        # replace the begin of doc with newline
        sed -e 's/ \[DOC\] \[SEP\] /\n/g' | \
        # handle the begin-symbol of the first doc
        sed -e 's/\[DOC\] \[SEP\] //g' | \
        # replace all [SEP] with </s>
        sed -e 's/\[SEP\]/<\/s>/g' > $prep/$F
    done
done
