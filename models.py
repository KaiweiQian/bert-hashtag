import torch
import torch.nn as nn

from transformers import BertModel, BertPreTrainedModel


class BertHashtag(nn.Module):
    def __init__(self):
        super(BertHashtag, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attn_mask):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attn_mask)

        cls_hidden = bert_out[0][:, 0, :]
        score = self.out(cls_hidden)

        return score


class BertHashtag1(BertPreTrainedModel):
    def __init__(self, config):
        super(BertHashtag1, self).__init__(config)
        self.bert = BertModel(config)
        self.out = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids, attn_mask):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attn_mask)

        cls_hidden = bert_out[0][:, 0, :]
        score = self.out(cls_hidden)

        return score
