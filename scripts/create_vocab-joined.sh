#!/bin/sh

python ZEST/preprocess.py -train_src /SiTS/spm/trans.si --train_tgt /SiTS/spm/trans.en --src_vocab_size 32000 --share_vocab --save_data data/sits