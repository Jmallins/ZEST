"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
import torch
from onmt.modules.util_class import Elementwise

fake = True
fake2 = False
class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerEncoder, self).__init__()

        if fake:
            self.spec = nn.Embedding(5, 512, padding_idx=embeddings.word_padding_idx, sparse=False) #,max_norm=3
            if False:
                self.mlp  = nn.Sequential(nn.Linear(514, 512))
                self.spec = nn.Embedding(5, 2, padding_idx=embeddings.word_padding_idx, sparse=False)

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.num_layers = num_layers
        if num_layers == 5:
            assert(False)
            self.tags = {"TRANS":0,"LM":1,"SIMPLY":2,"EN":3,"DE":4,"SIMP":5,"COMP":6}
            self.extras = nn.ModuleList()
            for k,v in self.tags.items():
                self.extras.append(TransformerEncoderLayer(
                    d_model, heads, d_ff, dropout,
                    max_relative_positions=max_relative_positions))
                if k=="SIMP" or k =="COMP":
                    continue
                self.extras.append(TransformerEncoderLayer(
                    d_model, heads, d_ff, dropout,
                    max_relative_positions=max_relative_positions))
        else:

            self.tags = {"TRANS":0,"LM":1,"SIMPLY":2,"EN":3,"DE":4,"SIMP":5,"COMP":6}
            self.extras = nn.ModuleList()
            for tag in self.tags:
            	self.extras.append(TransformerEncoderLayer(
                    d_model, heads, d_ff, dropout,
                    max_relative_positions=max_relative_positions))

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src,ctags, nograd=False, lengths=None):

        """See :func:`EncoderBase.forward()`"""
        assert("DE" not in ctags)
        assert("EN" not in ctags)
        
        markers2 = None
        if src.shape[2]==3:
            
            markers = src[:,:,-1:]

            if "SIMPLY" in ctags:
                #print ("HERE")
                markers2 = self.spec(markers).squeeze(dim=2).transpose(0, 1).contiguous()



            src = src[:,:,:-1]

         


        #print (src)
        #assert(False)

        if nograd:
        
            with torch.no_grad():
                self._check_args(src, lengths)
     
                emb = self.embeddings(src) 
                

                out = emb.transpose(0, 1).contiguous()

                words = src[:, :, 0].transpose(0, 1)
                w_batch, w_len = words.size()
                padding_idx = self.embeddings.word_padding_idx
                mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
                # Run the forward pass of every layer of the tranformer.
                for layer in self.transformer:
                    out = layer(out, mask)

        else:


            self._check_args(src, lengths)
 
            emb = self.embeddings(src)

            #emb = emb+ markers
            #print(emb.shape)
            #print (markers.shape)

            out = emb.transpose(0, 1).contiguous() 


            #print (out.shape)

            #print ("xxxx")
            words = src[:, :, 0].transpose(0, 1)
            w_batch, w_len = words.size()
            padding_idx = self.embeddings.word_padding_idx
            mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
            # Run the forward pass of every layer of the tranformer.
            for layer in self.transformer:
                out = layer(out, mask)
        #assert(len(ctags)>0)


        out2 = out
        #assert("EN" not in ctags)
        #assert("DE" not in ctags)
        for ii,t in enumerate(ctags):
            if t =="DE" or t=="EN":
                print (t)
                #assert(False)
            if markers2 is not None and t=="SIMPLY" and (ii ==0 or ctags[ii-1]!="SIMPLY"):
                if True:

                    out = out+markers2

                else:
                #print (out.shape)
                    markers3 = torch.cat((out,markers2),2)
                    #print (markers3.shape)
                    shapt =  (markers3.shape)
                    markers3 = markers3.reshape(-1,514)
                    markers3= self.mlp(markers3)
                    out = markers3.reshape(shapt[0],shapt[1],512)

                #emb_luts = Elementwise("mlp", markers3)
                
            if  t== "DE" or t =="EN":
                out2=out
            if self.num_layers!= 5 or ii == 0 or ctags[ii-1] != t:
                out = self.extras[self.tags[t]](out, mask)
            else:
                assert(False)
                out = self.extras[self.tags[t]+1](out, mask)




        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths,out.transpose(0, 1).contiguous() #2nd out should be out2



