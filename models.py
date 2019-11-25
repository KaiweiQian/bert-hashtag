import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class BertHashtag(BertPreTrainedModel):
    def __init__(self, config):
        super(BertHashtag, self).__init__(config)
        self.bert = BertModel(config)
        self.out = nn.Linear(768, 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attn_mask):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attn_mask)

        cls_hidden = bert_out[0][:, 0, :]
        print(cls_hidden.shape)
        score = self.out(cls_hidden)

        return score
