import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt
import random
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from onmt.Utils import formalize, deformalize, reorder


class Disc(nn.Module):
    """Auto Encoder class

    Attributes:
       size: input size
    """

    def __init__(self, d_input_size, hidden_size, embeddings, _):
        super(Disc, self).__init__()
        self.embeddings = embeddings
        self.src_rnn = nn.LSTM(d_input_size, hidden_size=hidden_size/2, dropout=0.2, bidirectional=True)
        self.tgt_rnn = nn.LSTM(d_input_size, hidden_size=hidden_size/2, dropout=0.2, bidirectional=True)
        self.ref_rnn = nn.LSTM(d_input_size, hidden_size=hidden_size/2, dropout=0.2, bidirectional=True)
        self.project1 = nn.Linear(hidden_size, 64)
        self.project2 = nn.Linear(64, 1)
        self.dp = nn.Dropout(p=0.2)
        self.act = nn.Sigmoid()

    def encode(self, obs, obs_lengths, hidden, rnn):
        r, r_len_sorted, origin_new = formalize(obs, obs_lengths)

        r_emb = self.embeddings(r)
        packed_r_emb = pack(r_emb, r_len_sorted.view(-1).tolist())

        if hidden is None:
            r_output, r_hidden = rnn(packed_r_emb)
        else:
            init_hidden = [reorder(x, origin_new) for x in hidden]
            r_output, r_hidden = rnn(packed_r_emb, init_hidden)
        r_output, _ = unpack(r_output)

        obs_context = deformalize(r_output, origin_new)
        obs_hidden = tuple([deformalize(x, origin_new) for x in r_hidden])
        return obs_hidden, obs_context, obs_lengths

    def forward(self, tgt, tgt_lengths, src, src_lengths, step_type='d_step'):
        """Feed forward process of seq2seq

        Args:
            input: Variable(LongTensor(batch_size, N)), N is the length of input sequence.
        Returns:
            list of decoded tensor
        """
        # encode src and obs
        src_emb = self.embeddings(src)
        packed_src_emb = pack(src_emb, src_lengths.tolist())
        src_output, src_hidden = self.src_rnn(packed_src_emb)

        tgt_emb = self.embeddings(tgt[1:])
        tgt_output, tgt_hidden = self.tgt_rnn(tgt_emb, src_hidden)

        logits = self.project1(tgt_output)
        logits = self.project2(self.dp(F.leaky_relu(logits)))
        logits = self.act(logits)
        if step_type == 'd_step':
            return torch.mean(logits, dim=0)

        return logits
