import torch

from json import load

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from models import BertSquad
from transformers import BertTokenizer


def load_squad_data(file_path):
    dict_data = load(file_path)


bert_type = 'bert-base-uncased'


tokenizer = BertTokenizer.from_pretrained(bert_type)

squad_model = BertSquad.from_pretrained(bert_type)
optimizer = Adam(squad_model.parameters())

start_score, end_score = squad_model(input_ids=input_ids, token_type_ids=token_type_ids)