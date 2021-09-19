# Repo

This repo contains the code used within the paper [Zero-Shot Crosslingual Sentence Simplification](http://link.to.come.later)


For the test data and GeoLino dataset see https://github.com/Jmallins/ZEST-data

# Instructions for use 

There are three steps when using ZEST: Data, Preprocessing, Training, and Predicition. 


## Data

Zest requires at a minimum: translation data and simplification data (in one language). Additionally, we experimented with the inclusion of non-parallel data, see table 5 for a comparison. 

Translation data can be taken from WMT (in the paper we used wmt19, but feel free to use the most recent WMT (2021). Additionally, you could consider using opensubtiles dataset or wikimatrix. 

For simplification data we used wikilarge, however, you could consider newsela, or more recent wikipedia simplification datasets https://github.com/chaojiang06/wiki-auto . For wikilarge please ensure you use the correction version (one version will have entities replaced with placeholders). Wikilarge will also need to be detokenized, mosses detokenization can be used. 

For non-parallel data we used simple wikipedia, geolino, and wmt data (see table 2). We used this for auto-encoding and a language modeling task.

# Steps

In the paper, we tokenized (use a reversible language-specific tokenizer) all our data and then learnt truecasing on all our data. This is time-consuming and probably offers no benefits, so this step could be skipped.

We next learnt sentencepiece, we learnt sentencepieceon all available data (non-parallel, simplification, and translation). This involves making one dataset. We learnt a sample based SP vocab of size 50K. Note, a large size SP vocab will significantly affect your training speeds, as such you could consider using 32K. 

The next step was source word dropout, this is important if you include non-parallel data, but could be skipped if you do not include this data (for the paper we always included this step). Drop 10% of the source tokens. Note, other noise functions could work.  For the paper, we upsampled all datasets to at least ~1M datapoints, such that different source word dropout is applied. 

Apply sentencepiece to the data. For the auto-encoding data we applied different samples of sentencepieces to the source and target.




## Preprocessing

First, create the vocab file (from English and German), all files should already be split into sentence pieces and ideally they should have the same tokenization (mosses tokenziation/detokenization could be used).

```
python preprocess.py -train_src X --train_tgt Y --share_vocab --save_data some_dir/the_vocab
```
Where X and Y are files that include all the sources and all the targets. You can also set the vocab size, which should equal the SP vocab size + 10 or so for special tokens. You can also set the source and target length. In the paper we used 128 for both, note that the longer the length the slower the training. See openNMT-py for additional documentation. 

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
Being Source language, task+, domain, output language. A layer (apart from languages) can be applied multiple times by including it multiple times in the key.  If you have simplification data in German it can also be included. Note, we use DE/EN but these are not language specific and any two languages could be used. Note, this json corrosponds to table 1. If you do not wish to include all the tasks, simply remove them from the json.
##  Training

To train a model 
```
python  train.py -data  some_dir\something.json  -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8         -encoder_type transformer -decoder_type transformer -position_encoding         -train_steps 2500000  -max_generator_batches 2 -dropout 0.1         -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2         -max_grad_norm 0 -param_init 0  -param_init_glorot         -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 --share_embeddings -warm_up 1000000 -cool_down  2400000 -lm-boost 0.5 -multi True 
```

Where some_dir/data.json is the json file created previously. Note, this could take over a week to train (on a tesla-p100), so its worth checking everything is correct (there is nothing worse than waiting a week for the data to be wrong, trust me). Checkpoints are saved regularly so check they do something sensible.  warm_up indicates the number of steps the model will only train on translation data,  after cool_down steps are exceded all tasks will be trained on evenly, and lm_boost (0-1) decrease the chances of the LM task being chosen. 

Important params are warm_up and train_steps.  You should aim for about 10 epochs of train_steps, note train steps is based on tokens. warm_up should be equal to about25\% to 50\% of training.


## Predicition 
```
python translate.py --length_penalty avg --beam_size 10  --replace_unk --dynamic_dict --share_vocab  --gpu 0 --model model_location.pt  -v --threshold 0 --ctags SIMPLY-SIMPLY-SIMPY-SIMP-DE  --batch_size 20  --max_length 128 --src input_file --output output_location 
```
Note, any combination of layers can be provided.  Also, please remember to do checkpoint selection, for speed reasons you might consider setting beam size to 1. If you notice your outputs are not simple enough you can apply the simplification layer multiple times to increase the simplicity.

## Notes and Regrets 

* We detokenized wikilarge with the joshua detokenization script
* We applied tokenization, truecasing and sentencepieces. If I was to do it again I would just apply sentencepieces.
* We sampled different sentencepieces for the source the target when autoencoding.
* We applied source word dropout before applying sentencepieces.
* We upsampled all data to ~1M examples, and applied different sentencepieces sampling and source word dropout.
* We learnt sentencepieces, truecasing, and vocab on all data. This is very timely and instead these could probably be learnt only on the translation data. 
* For the paper we did not use ParaCrawl from WMT19.
* Regret, the validation set (or some of it) could have been used as training source. 
