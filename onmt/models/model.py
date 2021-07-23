"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
      decoder2 (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder,decoder2=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder2 = decoder2

    def forward(self, src, tgt, lengths,tags=[],nograd=False, bptt=False,dumpenc=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            dumpenc: should the encoder states be returned. 

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        tgt = tgt[:-1]  # exclude last target from inputs

        if self.decoder2 is not None:
            lang = tags[-1]
            tags = tags[:-1]

        enc_state, memory_bank, lengths,memory_bank2 = self.encoder(src, tags,nograd,lengths)

        if self.decoder2 is not None:
            if  lang == "SI":
                if bptt is False:
                    self.decoder.init_state(src, memory_bank, enc_state)
                dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths)
            else:
           
                if bptt is False:
                        self.decoder2.init_state(src, memory_bank, enc_state)
                dec_out, attns = self.decoder2(tgt, memory_bank,
                                          memory_lengths=lengths)
        else:
            if bptt is False:
                self.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths)
        if dumpenc:
            return dec_out,attns,memory_bank2,memory_bank
        return dec_out, attns
