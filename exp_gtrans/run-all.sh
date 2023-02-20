#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 run-randinit train exp_test"
    echo "    bash $0 run-finetune train exp_test"
    echo "    bash $0 run-mbart train exp_test"
    exit
fi

RUN=exp_gtrans/$1.sh
bash $RUN iwslt17 $2 $3 $4 $5 $6 $7
bash $RUN nc2016 $2 $3 $4 $5 $6 $7
bash $RUN europarl7 $2 $3 $4 $5 $6 $7

