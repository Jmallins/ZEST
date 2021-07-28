#!/bin/sh

mkdir predictions
python ZEST/translate.py -block_ngram_repeat 5 --length_penalty avg --gpu 0 --beam_size 1 --random_sampling_topk 5 --random_sampling_temp 0.8 --batch_size 20 --max_length 128 --replace_unk --dynamic_dict --share_vocab --model saved_models/model_step_250006.pt -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMP-SI --src data/spm/sist.valid.src --output predictions/si-pred-rand-samp-1lyrs.txt
python ZEST/translate.py -block_ngram_repeat 5 --length_penalty avg --gpu 0 --beam_size 1 --random_sampling_topk 5 --random_sampling_temp 0.8 --batch_size 20 --max_length 128 --replace_unk --dynamic_dict --share_vocab --model saved_models/model_step_250006.pt -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMP-SI --src data/spm/sist.valid.src --output predictions/si-pred-rand-samp-2lyrs.txt
python ZEST/translate.py --length_penalty avg --beam_size 10  --replace_unk --dynamic_dict --share_vocab  --gpu 0 --model saved_models/model_step_250006.pt  -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMP-SI  --batch_size 20  --max_length 128 --src data/spm/sist.valid.src --output predictions/si-pred-beam-search.txt
