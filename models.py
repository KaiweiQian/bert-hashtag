import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class BertSquad(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSquad, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids)

        last_hidden = bert_out[0]

        scores = self.qa_outputs(last_hidden)
        start_score, end_score = scores.split(1, dim=-1)

        return start_score, end_score
