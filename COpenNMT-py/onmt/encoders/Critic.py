
import torch.nn as nn
from torch import functional as F
from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class Critic(nn.Module):

	    def __init__(num_layers, hidden_size, dropout=0.0):
	        super(Critic, self).__init__()


	        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])


	        for i in range(1,num_layers):
	        	self.linears.append(nn.Linear(hidden_size, hidden_size))



	        self.final = (nn.ModuleList([nn.Linear(hidden_size, 1)]))



	     def foward(self,x):


	     	for linear in self.linears:
	     		x = F.Relu(linear(x))


	     	return F.sigmoid(self.final(x))


class CriticLossCompute(LossComputeBase):
	
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CriticLossCompute, self).__init__(criterion, generator)
        #self.tgt_vocab = tgt_vocab
        #self.normalize_by_length = normalize_by_length


    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")
        assert(False)
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        }


    def _compute_loss(self, batch, output, target):
        """Compute the loss.
        The args must match :func:`self._make_shard_state()`.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """

        target = target.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if False:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats