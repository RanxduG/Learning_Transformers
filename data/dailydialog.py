import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class DialogueDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=200):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self.load_data(data_path)

    def load_data(self, path):
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if '__eou__' in line:
                    utterances = line.strip().split('__eou__')
                    utterances = [u.strip() for u in utterances if u.strip()]
                    for i in range(len(utterances) - 1):
                        src = utterances[i]
                        tgt = utterances[i + 1]
                        pairs.append((src, tgt))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src = self.tokenizer(src_text, padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors='pt')
        tgt = self.tokenizer(tgt_text, padding='max_length', truncation=True,
                             max_length=self.max_len, return_tensors='pt')

        return {
            'src_ids': src['input_ids'].squeeze(0),
            'src_mask': src['attention_mask'].squeeze(0),
            'tgt_ids': tgt['input_ids'].squeeze(0),
            'tgt_mask': tgt['attention_mask'].squeeze(0)
        }


def get_dataloader(data_path, batch_size, max_len=200):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = DialogueDataset(data_path, tokenizer, max_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer