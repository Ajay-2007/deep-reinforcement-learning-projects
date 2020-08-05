import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from . import utils

HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50


class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(
            num_embeddings=dict_size, embedding_dim=emb_size)

        self.encoder = nn.LSTM(
            input_size=emb_size, hidden_size=hid_size,
            num_layers=1, batch_first=True)

        self.decoder = nn.LSTM(
            input_size=emb_size, hidden_size=hid_size,
            num_layers=1, batch_first=True)

        self.output = nn.Linear(hid_size, dict_size)

    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index + 1].contiguous(), \
               encoded[1][:, index:index + 1].contiguous()

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x):
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        out = self.output(out)

        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len,
                            stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        :param hid:
        :param begin_emb:
        :param seq_len:
        :param stop_at_token:
        :return:
        """
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_logits, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)

            res_logits.append(out_logits)
            res_tokens.append(out_token)

            if stop_at_token is not None:
                if out_token == stop_at_token:
                    break

        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len,
                              stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        :param hid:
        :param begin_emb:
        :param seq_len:
        :param stop_at_token:
        :return:
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hig = self.decode_one(hid, cur_emb)
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]

            action = int(np.random.choice(
                out_probs.shape[0], p=out_probs
            ))

            action_v = torch.LongTensor([action])
            action_v = action_v.to(begin_emb.device)
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None:
                if action == stop_at_token:
                    break

        return torch.cat(res_logits), res_actions


def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)

    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)

    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x

    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(
        input_v, lens, batch_first=True
    )
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(
        r, input_seq.batch_sizes
    )

    return emb_input_seq, input_idx, output_idx
