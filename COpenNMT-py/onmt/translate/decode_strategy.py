#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import torch
import numpy as np
import pyphen
class DecodeStrategy(object):
    """Base class for generation strategies.

    Args:
        pad (int): Magic integer in output vocab.
        bos (int): Magic integer in output vocab.
        eos (int): Magic integer in output vocab.
        batch_size (int): Current batch size.
        device (torch.device or str): Device for memory bank (encoder).
        parallel_paths (int): Decoding strategies like beam search
            use parallel paths. Each batch is repeated ``parallel_paths``
            times in relevant state tensors.
        min_length (int): Shortest acceptable generation, not counting
            begin-of-sentence or end-of-sentence.
        max_length (int): Longest acceptable sequence, not counting
            begin-of-sentence (presumably there has been no EOS
            yet if max_length is used as a cutoff).
        block_ngram_repeat (int): Block beams where
            ``block_ngram_repeat``-grams repeat.
        exclusion_tokens (set[int]): If a gram contains any of these
            tokens, it may repeat.
        return_attention (bool): Whether to work with attention too. If this
            is true, it is assumed that the decoder is attentional.

    Attributes:
        pad (int): See above.
        bos (int): See above.
        eos (int): See above.
        predictions (list[list[LongTensor]]): For each batch, holds a
            list of beam prediction sequences.
        scores (list[list[FloatTensor]]): For each batch, holds a
            list of scores.
        attention (list[list[FloatTensor or list[]]]): For each
            batch, holds a list of attention sequence tensors
            (or empty lists) having shape ``(step, inp_seq_len)`` where
            ``inp_seq_len`` is the length of the sample (not the max
            length of all inp seqs).
        alive_seq (LongTensor): Shape ``(B x parallel_paths, step)``.
            This sequence grows in the ``step`` axis on each call to
            :func:`advance()`.
        is_finished (ByteTensor or NoneType): Shape
            ``(B, parallel_paths)``. Initialized to ``None``.
        alive_attn (FloatTensor or NoneType): If tensor, shape is
            ``(step, B x parallel_paths, inp_seq_len)``, where ``inp_seq_len``
            is the (max) length of the input sequence.
        min_length (int): See above.
        max_length (int): See above.
        block_ngram_repeat (int): See above.
        exclusion_tokens (set[int]): See above.
        return_attention (bool): See above.
        done (bool): See above.
    """

    def __init__(self, pad, bos, eos, batch_size, device, parallel_paths,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length,i2w,batch,threshold,sent1,badsynf):

        # magic indices
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.i2w = i2w
        self.batch=batch
        self.cache = {}
        self.threshold = threshold
        self.sent1 = -sent1
        self.badsyn = None
        self.badparts = []
        from collections import defaultdict

        self.seen = defaultdict(set)
        if not badsynf is None:
            self.badsyn = set()

            f = open(badsynf)

            for line in f:
                line = line.strip()
                line = line.split()
                line = sorted(line)
                line = tuple(line)
                self.badsyn.add(line)
                #print (self.badsyn)
            f.close()

        self.ignore = set(["ROOT(",")ROOT"]  + ['LEN5', 'LEN9', 'CONJ', 'LEN4', 'EXPL', 'APPOS', 'ADVCL', 'IOBJ', 'DET', 'xDIVx', 'CASE', 'NMOD', 'NUMMOD', 'AUX:PASS', 'AMOD', 'OBL', 'OBJ', 'MARK', 'NSUBJ', 'LEN7', 'FLAT', 'NSUBJ:PASS', 'LEN8', 'LEN3', 'AUX', 'COP', 'LEN0', 'COMPOUND:PRT', 'LEN6', 'ADVMOD', 'PUNCT', 'LEN1', 'DEP', 'SPLIT', 'ACL', 'CC', 'LEN2'])




        # result caching
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]

        self.alive_seq = torch.full(
            [batch_size * parallel_paths, 1], self.bos,
            dtype=torch.long, device=device)
        self.is_finished = torch.zeros(
            [batch_size, parallel_paths],
            dtype=torch.uint8, device=device)
        self.alive_attn = None

        self.min_length = min_length
        self.max_length = max_length
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens
        self.return_attention = return_attention

        self.dic = pyphen.Pyphen(lang='de_DE')

        self.done = False
        self.setup()
    
    

    def setup(self):
        from sklearn.preprocessing import PolynomialFeatures
        import pickle
        self.mins= 1.0311042920747262e-07
        self.minc =2.1013035013908872e-10
        self.minc2 = 1.2799770628110345e-07
        self.poly = PolynomialFeatures(2,interaction_only=False)
        f = open("/home/s1564225/notebook/cwi.germa.p","rb")
        (lr,simpleC2,newsD2,complexC2) = pickle.load(f)
        f.close()
        self.lr = lr
        self.simpleC2=simpleC2
        self.newsD2=newsD2
        self.complexC2=complexC2


    def __len__(self):
        return self.alive_seq.shape[1]

    def ensure_min_length(self, log_probs):
        if len(self) <= self.min_length:
            log_probs[:, self.eos] = -1e20

    def ensure_max_length(self):
        # add one to account for BOS. Don't account for EOS because hitting
        # this implies it hasn't been found.
        if len(self) == self.max_length + 1:
            self.is_finished.fill_(1)

    def block_ngram_repeats(self, log_probs):
        cur_len = len(self)
        if self.block_ngram_repeat > 0 and cur_len > 1:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOS
                hyp = self.alive_seq[path_idx, 1:]
                ngrams = set()
                fail = False
                gram = []
                for i in range(cur_len - 1):
                    # Last n tokens, n = block_ngram_repeat
                    gram = (gram + [hyp[i].item()])[-self.block_ngram_repeat:]
                    # skip the blocking if any token in gram is excluded
                    if set(gram) & self.exclusion_tokens:
                        continue
                    if tuple(gram) in ngrams:
                        fail = True
                    ngrams.add(tuple(gram))
                if fail:
                    log_probs[path_idx] = -10e10 + log_probs[path_idx]


    def complexword(self,word):
        if self.threshold ==0:
            return False
       
        #print (self.threshold)

        score = self.complexscore(word)
        if score[0] > self.threshold:
            return True
        return False

    def complexscore(self,word):
        def simpprob(word):
            word = word.lower()
            if word in self.simpleC2:
                return self.simpleC2[word]
            else:
                return self.mins
            
        def compprob(word):
            word = word.lower()
            if word in self.newsD2:
                return self.newsD2[word]
            else:
                return self.minc
            
        def compprob2(word):
            word = word.lower()
            if word in self.complexC2:
                return self.complexC2[word]
            else:
                return self.minc2

        def vowelcount(word):
            vowels = ["a","o","u","i","ä","ö","ü"]
            count = 0
            for v in vowels:
                count+=word.count(v)
            return count






        def sylcount(word):
            word2 = word.replace("-","")
            return len(self.dic.inserted(word2).split("-"))

        def wordcount(word):
            return len(word)
                    
        multi = "-" in word and word[-1]!="-" and word[0]!="-"
        feat = [np.log(compprob2(word)),np.log(simpprob(word)),np.log(compprob(word)),vowelcount(word),sylcount(word),wordcount(word),multi]

        return  self.lr.predict(self.poly.fit_transform(np.array(feat).reshape(1, -1)))


    def bannedtemplates(self,hyps):
  
        current = []
        bads = self.badsyn
        #
        start = False
        for h in  hyps.split():

            if h ==")root":

                start = False
                if tuple(sorted(current)) in bads:
            
                    return True
                current = []
                continue
            if h=="root(":
                start = True
                continue
            if start:
                current.append(h)
        return False


    def forcepath(self,hyps,paths):

        start = False
        current = []

        paths = set(paths)
        for h in  hyps.split():
            if h=="root(":
                start = True
                continue
            if h ==")root":
                current.append(h)
                start = False
                #print (sorted(current))
                if tuple(sorted(current))  not in paths:
                    return True
            
                
             
                current = []
                continue
            if start:
                current.append(h)

                if tuple(sorted(current))  not in paths:

                    return True
      
        return False




    def bannedbpe(self,log_probs,forced=[],batch_offset=[]):
        from collections import defaultdict
        if True and ( self.sent1 ==0 and self.threshold==0 and 0 == sum([len(q) for q in forced]) and (self.badsyn is None or self.badsyn ==[])):
          
            return 



        if False and self.sent1 !=0:
            for path_idx in range(self.alive_seq.shape[0]):

                alltemps = []
                current = []
                start = False
                for w in forced[path_idx][1:-1]:
                    w = self.i2w[w]
  
                    if w =="ROOT(":
                        start = True
                        #current.append(w)
                        continue
                    if start and w == ")ROOT":
                        #current.append(w)
                        start = False
                        alltemps.append(current)
                        current = []
                        continue
                    if start:

                        current.append(w)
                        alltemps.append(tuple(sorted(current)))
                hyp = self.alive_seq[path_idx, 1:]
                hyps =  (" ".join([self.i2w[x] for x in hyp])).replace(" ","")
                if len(hyp) ==1 and hyps == "LEN1" and  alltemps== []:
                    
                    log_probs[path_idx] = log_probs[path_idx] +self.sent1

        #if self.threshold ==0:
        #    return
        #print ("xxxxx")
        nextc = 0
        today = defaultdict(set)
        for path_idx in range(self.alive_seq.shape[0]):
            if path_idx % self.beam_size == 0 and path_idx!=0:
                nextc+=1
                #print ("YYYY")
            ignore =  (set(" ".join([self.i2w[x].lower() for x in forced[path_idx][1:-1]]).replace(" ","").replace("▁"," ").split() ))
            alltemps = []
            current = []
            start = False

            values, indices = torch.max(log_probs[path_idx], 0)
            if float(values) < -100:

                continue
            

          

            for w in forced[path_idx][1:-1]:
                w = self.i2w[w].lower()

                if w =="root(":
                    start = True
                    #current.append(w)
                    continue
                if start and w == ")root":
                    current.append(w)
                    start = False
                    alltemps.append(tuple(sorted((current))))
                    current = []
                    continue
                if start:

                    current.append(w)
                    alltemps.append(tuple(sorted(current)))

            gignore = ignore.difference(self.ignore)
            ignore = ignore.union(self.ignore)
            hyp = self.alive_seq[path_idx, 1:]
            if len(hyp) < 1:
                continue
            hyps =  (" ".join([self.i2w[x] for x in hyp])).replace(" ","").lower()


            if len(hyp) ==1 and self.sent1 !=0 and  alltemps== []  and hyps == "len1":
  
                log_probs[path_idx] = log_probs[path_idx] +self.sent1
            #print ([self.i2w[x] for x in hyp])

            if ")root" in hyps:
                #print (hyps)
                #assert(False)
                hypsrs = hyps.split(")root")
                hypsR = hypsrs[0]
                hypsRl = len(hypsR+")root")

                hypsR = sorted(hypsR.split())
                hypsR = " ".join(hypsR)
                hypsrs[0]= hypsR
                hyps = ")root".join(hypsrs)


            if hyps in self.seen[int(batch_offset[nextc])]:
                log_probs[path_idx] = -10e10
                continue




            self.seen[int(batch_offset[nextc])].add(hyps)
            today[int(batch_offset[nextc])].add(hyps)
  



            if  alltemps== [] and self.badsyn is not None  and self.bannedtemplates(" ".join([self.i2w[x] for x in hyp]).lower()):
                log_probs[path_idx] = -10e10 

            if True and alltemps!= [] and self.forcepath(" ".join([self.i2w[x] for x in hyp]).lower(),alltemps):

                log_probs[path_idx] = -10e10 


            
      
            if "▁"  in hyps and not hyps[-1]=="▁" and True:
                found = False
                if "▁" in hyps: 
                    lwd = hyps.split("▁")[-1]
                    if "xdivx" in lwd:
                        lwd = lwd.split("xdivx")[-1]
                    if "split" in lwd:
                        lwd = lwd.split("split")[-1]
                    if lwd.strip()=="":
                        found = True
   
                    for iii in gignore:

                        if iii.lower().startswith(lwd):
                            found = True
                            break
                if not found:
                    values, indices = torch.max(log_probs[path_idx], 0)
                    #print (indices)
                    #print (values)
                    log_probs[path_idx] = -10e10
                    log_probs[path_idx][int(indices)]= values
            if "▁" in hyps:

                lwd = hyps.split("▁")[-2].lower()
                #print (lwd)
                if "xdivx" in lwd:
                    lwd = lwd.split("xdivx")[-1]
                if lwd =="":
                    continue
                if lwd in ignore:
                    continue

   
                
                if self.complexword(lwd) and True:
          
                    log_probs[path_idx] = -10e10
                    continue


        nextc = 0

        for path_idx in range(self.alive_seq.shape[0]):
            if path_idx % self.beam_size == 0 and path_idx!=0:
                nextc+=1

            values, indices = torch.max(log_probs[path_idx], 0)
            if float(values) < -100:
            
                continue
               

            hyp = self.alive_seq[path_idx, 1:]
            if len(hyp) < 2:
                continue
            hyps =  (" ".join([self.i2w[x] for x in hyp])).replace(" ","").lower()

            if len(hyp) > 5 and False:
                tq = " ".join([self.i2w[x] for x in hyp[-5:]])
               
                if "▁" not in tq and not "len" in hyps:
                    log_probs[path_idx] = -10e10 
                    continue
                if "len" in hyps and "xdivx" in hyps:
                    if "▁" not in tq and not hyps.split()[-1].isupper() and not ")" in hyps and ("(") not in tq : 
                        log_probs[path_idx] = -10e10
                        continue
            if not "▁" in hyps:
                continue
            if   True:
                #print (len(self.seen[int(batch_offset[nextc])]))
               
                lwdll = hyps
                for s in self.seen[int(batch_offset[nextc])]:
                    #print (s)
                    if s in today[int(batch_offset[nextc])]:
                        continue
                    if  s.startswith(lwdll):
            

                        log_probs[path_idx] = -10e10

        return 



        ####################################################################################################
        #seen = set()
      
        self.badparts = defaultdict(list)
        
        nextc = 0
        #print (self.seen)
        #print (self.seen)
        #print ("XXX")
        
        for path_idx in range(self.alive_seq.shape[0]):
            if path_idx % self.beam_size == 0 and path_idx!=0:
                nextc+=1
            #print (forced)
            ignore =  (set(" ".join([self.i2w[x].lower() for x in forced[path_idx][1:-1]]).replace(" ","").replace("▁"," ").split() ))
            ignore = ignore.union(self.ignore)
            hyp = self.alive_seq[path_idx, 1:]
            hyps =  (" ".join([self.i2w[x] for x in hyp])).replace(" ","").lower()
            #print (" ".join([self.i2w[x] for x in hyp]))
            if ")root" in hyps:
                #print (hyps)
                #assert(False)
                hypsrs = hyps.split(")root")
                hypsR = hypsrs[0]
                hypsRl = len(hypsR+")root")

                hypsR = sorted(hypsR.split())
                hypsR = " ".join(hypsR)
                hypsrs[0]= hypsR
                hyps = ")root".join(hypsrs)

            if len(hyp) > 2 and (hyps) in self.seen[int(batch_offset[nextc])]:
              
                log_probs[path_idx] = -10e10 + log_probs[path_idx] 
                continue
            else:
                self.seen[int(batch_offset[nextc])].add(hyps)
                today[int(batch_offset[nextc])].add(hyps)
            if "▁" in hyps:

                lwd = hyps.split("▁")[-2].lower()
                #print (lwd)
                if "xdivx" in lwd:
                    lwd = lwd.split("xdivx")[-1]
                if lwd =="":
                    continue
                if lwd in ignore:
                    continue

                if hyps.startswith("LEN") and "xDIVx" not in hyps:
           
                    continue
                
                if self.complexword(lwd) and True:
                    print (lwd)
                    #print ("BAD")
                    #print (lwd)

                    self.badparts[int(batch_offset[nextc])].append(lwd)
                    log_probs[path_idx] = -10e10 + log_probs[path_idx] 
        nextc = 0

        for path_idx in range(self.alive_seq.shape[0]):
            if path_idx % self.beam_size == 0 and path_idx!=0:
                nextc+=1

            scache = tuple( forced[path_idx])
            if False and scache in self.cache:
                ignore,alltemps = self.cache[scache]
            else:
                alltemps = []
                current = []
                start = False
                for w in forced[path_idx][1:-1]:
                    w = self.i2w[w]
  
                    if w =="ROOT(":
                        start = True
                        #current.append(w)
                        continue
                    if start and w == ")ROOT":
                        #current.append(w)
                        start = False
                        alltemps.append(current)
                        current = []
                        continue
                    if start:

                        current.append(w)
                        alltemps.append(tuple(current))


      
                #print (ignore)

                self.cache[scache]=(ignore,alltemps)

           

            #print (ignore)
            hyp = self.alive_seq[path_idx, 1:]
            if len(hyp) ==0:
                continue
            li = hyp[-1]
            lw = self.i2w[hyp[-1]]
            if li ==self.pad or li ==self.eos:
  
                continue

            hyps = " ".join([self.i2w[x] for x in hyp])
            if len(hyp) ==1 and hyps == "LEN1" and  alltemps== []:
                log_probs[path_idx] = log_probs[path_idx] +self.sent1

            if   alltemps== [] and self.badsyn is not None  and self.bannedtemplates(hyps):
                log_probs[path_idx] = -10e10 + log_probs[path_idx]

            if alltemps!= [] and self.forcepath(hyps,alltemps):
                log_probs[path_idx] = -10e10 + log_probs[path_idx]


            if len(hyp) > 5 and True:
                tq = " ".join(hyps.split()[-5:])
                if "▁" not in tq and not "LEN" in hyps:
                    log_probs[path_idx] = -10e10 + log_probs[path_idx]
                if "LEN" in hyps and "xDIVx" in hyps:
                    if "▁" not in tq and not hyps.split()[-1].isupper() and not ")" in hyps and ("(") not in tq : 
                        log_probs[path_idx] = -10e10 + log_probs[path_idx]



            lwd = (hyps.replace(" ","")).strip()

            if len(hyp) > 2 and  True:
                #print (len(self.seen[int(batch_offset[nextc])]))
             
                lwdll = lwd.lower()
                for s in self.seen[int(batch_offset[nextc])]:
                    #print (s)
              
                    if  s.startswith(lwdll) and s!=lwdll:
         

                        log_probs[path_idx] = log_probs[path_idx]  +-10e10

          
            lwd = lwd.split("xDIVx")[-1].replace("SPLIT","")
            #print (lwd)

            if lwd.startswith("LEN") and "xDIVx" not in lwd:
           
                continue

            if "▁" not in lwd:
        
                continue


            #print (lwd)
            if  lwd.split("▁")[-1] !="" and True:
                #print (lwd)
                #print (self.badparts)
                lwdt = lwd.split("▁")[-1]
                for w in self.badparts[int(batch_offset[nextc])]:
                    if w =="":
                        continue
                    if  w.startswith(lwdt):
                        #print (w)
                        #print (lwdt)
                        #print ()
                        log_probs[path_idx] = -10e10 + log_probs[path_idx]






    def force(self,log_probs,forced):
        cur_len = len(self)
        #print (self.alive_seq.shape[0])
        #print (forced)
        return
        #print (forced)
        if cur_len>= max([len(x) for x in [forced]])-1:
            return

        if forced is not None and forced !=[]:
            for path_idx in range(self.alive_seq.shape[0]):
                # skip BOSs
                hyp = self.alive_seq[path_idx, 1:]
                fail = False
                gram = []
         
                forceds  = forced[path_idx]

                if cur_len <len(forceds)-1:

   
                    #backup = log_probs[path_idx][forceds[cur_len]]
                    log_probs[path_idx] = -10e20
                    log_probs[path_idx][forceds[cur_len]] = 0

       


    def advance(self, log_probs, attn):
        """DecodeStrategy subclasses should override :func:`advance()`.

        Advance is used to update ``self.alive_seq``, ``self.is_finished``,
        and, when appropriate, ``self.alive_attn``.
        """

        raise NotImplementedError()

    def update_finished(self):
        """DecodeStrategy subclasses should override :func:`update_finished()`.

        ``update_finished`` is used to update ``self.predictions``,
        ``self.scores``, and other "output" attributes.
        """

        raise NotImplementedError()

