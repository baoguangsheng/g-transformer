#!/usr/bin/env python3 -u
# Guangsheng Bao: changed on 2020/10/3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from fairseq_cli.preprocess import cli_main

if __name__ == '__main__':
    cli_main()
