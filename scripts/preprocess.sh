#!/bin/sh

python ZEST/preprocess.py -train_src /SiTS/spm/spm/si-en.en -train_tgt /SiTS/spm/si-en.si  -save_data data/EN-TRANS-COMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-en.si -train_tgt /SiTS/spm/si-en.en  -save_data data/SI-TRANS-COMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-en.en -train_tgt /SiTS/spm/si-en.en -save_data data/EN-TRANS-COMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/simpleWiki.en -train_tgt /SiTS/spm/simpleWiki.en  -save_data data/EN-TRANS-SIMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-en.si -train_tgt /SiTS/spm/si-en.si  -save_data data/SI-TRANS-COMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-100k.si -train_tgt /SiTS/si-100k.si  -save_data data/SI-TRANS-SIMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-en.en -train_tgt /SiTS/spm/si-en.en -save_data data/EN-LM-COMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/simpleWiki.en -train_tgt /SiTS/spm/simpleWiki.en  -save_data data/EN-LM-SIMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-en.si -train_tgt /SiTS/spm/si-en.si  -save_data data/SI-LM-COMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/si-100k.si -train_tgt /SiTS/spm/si-100k.si  -save_data data/SI-LM-SIMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/wikilarge.src -train_tgt /SiTS/spm/wikilarge.tgt -save_data data/EN-SIMPLY-SIMP-EN --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt

python ZEST/preprocess.py -train_src /SiTS/spm/sist.valid.src -train_tgt /SiTS/spm/sist.valid.tgt  -save_data data/valid-SI-SIMPLY-SIMP-SI --src_vocab data/sits.vocab.pt --tgt_vocab data/sits.vocab.pt