#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

path=egs/debug/tr
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python3 -m denoiser.audio $1 > $path/noisy.json
python3 -m denoiser.audio $2 > $path/clean.json
