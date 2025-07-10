import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder, generate_subsequent_mask


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)  # logits
        

class TransformerChatModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, ffn_hidden=2048, num_heads=8,
                 num_layers=6, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, num_layers, dropout, max_seq_len)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, num_layers, dropout, max_seq_len)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, tgt_mask=None):
        # src_ids: [B, src_len], tgt_ids: [B, tgt_len]
        src_embed = self.embedding(src_ids)
        tgt_embed = self.embedding(tgt_ids)

        memory = self.encoder(src_embed)
        out = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        return self.generator(out)