#!/bin/sh
mkdir data
python ZEST/preprocess.py -train_src SiTS/spm/si-en.si --train_tgt SiTS/spm/si-en.en --src_vocab_size 32000 --share_vocab --save_data data/sits