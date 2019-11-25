import json
import torch

from transformers import *
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, file_path, meta_path, max_len):
        with open(file_path) as f:
            raw_data = json.load(f)

        with open(meta_path) as f:
            meta = json.load(f)

        idx_data = create_idx(raw_data, max_len)
        del raw_data

        self.n = len(idx_data['id'])
        self.data = {}

        for i in range(self.n):
            self.data[idx_data['id'][i]] = (idx_data['token_ids'][i],
                                            idx_data['token_type_ids'][i],
                                            idx_data['attention_mask'][i],
                                            meta[idx_data['label'][i]]
                                            )

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        token_ids = torch.Tensor(self.data[index][0])
        token_type_ids = torch.Tensor(self.data[index][1])
        attention_mask = torch.Tensor(self.data[index][2])
        label = torch.Tensor(self.data[index][3])

        return token_ids, token_type_ids, attention_mask, label


def create_idx(dict_data, max_len=256):
    assert len(dict_data['text']) == len(dict_data['label'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    output = {'id': dict_data['id'],
              'token_ids': [],
              'token_type_ids': [],
              'attention_mask': [],
              'label': dict_data['label']
              }

    n = len(dict_data['text'])
    for i in range(n):
        senA = '[CLS] ' + dict_data['text'][i] + ' [SEP]'

        token_ids = tokenizer.encode(senA)
        token_ids = token_ids + [0 for _ in range(max_len - len(token_ids))]

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids)
        attention_mask = [0 if elem != 0 else 1 for elem in token_ids]

        output['token_ids'].append(token_ids)
        output['token_type_ids'].append(token_type_ids)
        output['attention_mask'].append(attention_mask)
    return output