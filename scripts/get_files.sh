#!/bin/sh

wget https://dms.uom.lk/s/XjwkC9QyDoA2oqr/download
unzip download
unzip SiTS/si_en_74k-20210606T151953Z-001.zip
rm -rf SiTS/si_en_74k-20210606T151953Z-001.zip

wget https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2
tar -xf data-simplification.tar.bz2
rm -rf data-simplification.tar.bz2

mv si_en_74k/parallel-08.11.2020-Tr74K.si-en.si SiTS/si-en.si
mv si_en_74k/parallel-08.11.2020-Tr74K.si-en.en SiTS/si-en.en
rm -rf si_en_74k

mv data-simplification/wikilarge/wiki.full.aner.train.src SiTS/wikilarge.src
mv data-simplification/wikilarge/wiki.full.aner.train.dst SiTS/wikilarge.tgt
rm -rf data-simplification

mv SiTS/sinhala_simp_100000.txt SiTS/si-100k.si
mv SiTS/complex-1000.txt SiTS/sist.valid.src
mv SiTS/simp1000-piyumika.txt SiTS/sist.valid.tgt