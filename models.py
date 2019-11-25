import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class BertHashtag(BertPreTrainedModel):
    def __init__(self, config):
        super(BertHashtag, self).__init__(config)
        self.bert = BertModel(config)
        self.out = nn.Linear(config.hidden_size, 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask)

        last_hidden = bert_out[0]

        scores = self.out(last_hidden)
        start_score, end_score = scores.split(1, dim=-1)

        return start_score, end_score
