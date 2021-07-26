import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=trans.en,trans.si,simpleWiki.en,si-100k.si,sist.valid.src,sist.valid.tgt,wikilarge.src,wikilarge.tgt --model_prefix=spm --vocab_size=32000 --model_type=bpe')

sp = spm.SentencePieceProcessor()
sp.load('spm.model')

file_list = ["trans.en","trans.si","simpleWiki.en","si-100k.si","sist.valid.src","sist.valid.tgt","wikilarge.src","wikilarge.tgt"]

for infile in file_list:
  with open(infile, 'r') as rf, open('spm/'+ infile, 'w') as wf:
      for line in rf:
          wf.write(' '.join(sp.encode(line, out_type=str)))
          wf.write("\n")