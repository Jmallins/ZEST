# Repo

This repo contains the code used within the paper [Zero-Shot Crosslingual Sentence Simplification](http://link.to.come.later)


For the test data and GeoLino dataset see https://github.com/Jmallins/ZEST-data

# Instructions for use 

There are three steps when using ZEST: Preprocessing, Training, and Predicition. 


## Preprocessing

First, create the vocab file (from English and German), all files should already be split into sentence pieces and ideally they should have the same tokenization (mosses tokenziation/detokenization could be used).

```
python preprocess.py -train_src X --train_tgt Y --share_vocab --save_data some_dir/the_vocab
```

Second, create the training file per task/domain/source_language/target_language. All files should already be split into sentence pieces and ideally they should have the same tokenization (mosses tokenziation/detokenization could be used, wikilarge comes pre-tokenized), noise can also be applied before using preprocess.
```
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt  -save_data data/demo --src_vocab some_dir/the_vocab --tgt_vocab some_dir/the_vocab 
```

Third, create a json file that lists all of the training files and the source language-task-domain-target language.

```
{
"SOURCE_language-TASK-DOMAIN-TARGET_LANGUAGE":"file_location.train",
  "EN-TRANS-COMP-DE": "file_location.train",
  "DE-TRANS-COMP-EN": "file_location.train",
  "EN-TRANS-COMP-EN": "file_location.train",
  "EN-TRANS-SIMP-EN": "file_location.train",
  "DE-TRANS-COMP-DE": "file_location.train",
  "DE-TRANS-SIMP-DE": "file_location.train",
  "EN-LM-COMP-EN": "file_location.train",
  "EN-LM-SIMP-EN": "file_location.train",
  "DE-LM-COMP-DE": "file_location.train",
  "DE-LM-SIMP-DE": "file_location.train",
  "EN-SIMPLY-SIMPLY-SIMP-EN":"file_location.train",
  "vocab":"file_location",
  "valid-DE-SIMPLY-SIMPLY-SIMPLY-SIMP-DE":"file_location.train"
}

```
Being Source language, task+, domain, output language. A layer (apart from languages) can be applied multiple times by including it multiple times in the key. 
##  Training

To train a model 
```
python  train.py -data  some_dir\something.json  -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8         -encoder_type transformer -decoder_type transformer -position_encoding         -train_steps 250000  -max_generator_batches 2 -dropout 0.1         -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 1         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2         -max_grad_norm 0 -param_init 0  -param_init_glorot         -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 --share_embeddings -warm_up 100000 -cool_down  240000 -lm_boost 0.5
```

Where some_dir/data.json is the json file created previously. Note, this could take over a week to train (on a tesla-p100), so its worth checking everything is correct (there is nothing worse than waiting a week for the data to be wrong, trust me). Checkpoints are saved regularly so check they do something sensible. If you wish to train for a short time you should change the two hardcoded values in (220000) onmt/trainerC.py to the numebr of steps you choose.  warm_up indicates the number of steps the model will only train on translation data,  after cool_down steps are exceded all tasks will be trained on evenly, and lm_boost (0-1) decrease the chances of the LM task being chosen. 

## Predicition 
```
python translate.py --length_penalty avg --beam_size 10  --replace_unk --dynamic_dict --share_vocab  --gpu 0 --model model_location.pt  -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMPY-SIMP-DE  --batch_size 20  --max_length 128 --src input_file --output output_location 
```
Note, any combination of layers can be provided. 

## Notes and Regrets 

* We detokenized wikilarge with the joshua detokenization script
* We applied tokenization, truecasing and sentencepieces. If I was to do it again I would just apply sentencepieces.
* We sampled different sentencepieces for the source the target when autoencoding.
* We applied source word dropout before applying sentencepieces.
* We upsampled all data to ~1M examples, and applied different sentencepieces sampling and source word dropout.
* We learnt sentencepieces, truecasing, and vocab on all data. This is very timely and instead these could probably be learnt only on the translation data. 
* For the paper we did not use ParaCrawl from WMT19.
* Regret, the validation set (or some of it) could have been used as training source. 
