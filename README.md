# Repo

This repo contains the code used within the paper [Zero-Shot Crosslingual Sentence Simplification](http://link.to.come.later)


For the test data and GeoLino dataset see https://github.com/Jmallins/ZEST-data

# Instructions for use 

There are three steps when using ZEST: Preprocessing, Training, and Predicition. 


## Preprocessing

First, create the vocab file (from English and German), all files should already be split into sentence pieces and ideally they should have the same tokenization (mosses tokenziation/detokenization could be used).

```
python preprocess.py -train_src X --train_tgt Y --share_vocab True --save_data some_dir/the_vocab
```

Second, create the training file per task/domain/source_language/target_language. All files should already be split into sentence pieces and ideally they should have the same tokenization (mosses tokenziation/detokenization could be used).
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
  "vocab":"file_location"
  "valid-DE-SIMPLY-SIMPLY-SIMPLY-SIMP-DE":"file_location.train"
}

```
Being Source language, task+, domain, output language.

##  Training

To train a model 
```
python  train.py -data  some_dir\something.json  -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 8 -heads 12         -encoder_type transformer -decoder_type transformer -position_encoding         -train_steps 250000  -max_generator_batches 2 -dropout 0.1         -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 1         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2         -max_grad_norm 0 -param_init 0  -param_init_glorot         -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 --share_embeddings
```

Where some_dir/data.json is the json file created previously.

## Predicition 
```
python translate.py --length_penalty avg --beam_size 30  --replace_unk --dynamic_dict --share_vocab  --gpu 0 --model model_location.pt  -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMPY-SIMP-DE  --batch_size 20  --max_length 128 --src input_file --output output_location 
```
```
