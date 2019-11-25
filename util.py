import json
import torch

from transformers import *
from torch.utils.data import Dataset
from gc import collect


class TweetDataset(Dataset):
    def __init__(self, file_path, max_len):
        raw_data = json.load(file_path)

        idx_data = create_idx(raw_data, max_len)
        del raw_data

        self.n = len(idx_data['id'])
        self.data = {}

        for i in range(n):
            self.data[idx_data['id'][i]] = (idx_data['token_ids'][i],
                                            idx_data['token_type_ids'][i],
                                            idx_data['attention_mask'][i],
                                            idx_data['label'][i]
                                            )

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        token_ids = torch.tensor(self.data[index][0])
        token_type_ids = torch.tensor(self.data[index][1])
        attention_mask = torch.tensor(self.data[index][2])
        label = torch.tensor(self.data[index][3])

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