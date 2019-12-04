import torch.nn as nn

from transformers import BertModel


class BertHashtag(nn.Module):
    def __init__(self, num_class=2):
        super(BertHashtag, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(768, num_class)

    def forward(self, input_ids, token_type_ids, attn_mask):
        bert_out = self.bert(input_ids=input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attn_mask)
        pooled = bert_out[1]

        score = self.out(pooled)

        return score
