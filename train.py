import torch

from util import TweetDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models import BertHashtag


if __name__ == '__main__':
    data = TweetDataset(file_path='./data/train.txt',
                        meta_path='./data/meta.txt',
                        max_len=256)
    tweet_model = BertHashtag.from_pretrained('bert-base-uncased')
    optimizer = Adam(tweet_model.parameters(), lr=5e-4)

    train_dataloader = DataLoader(data, batch_size=32, shuffle=True)

    for it, batch in enumerate(train_dataloader):
        token_ids, token_type_ids, attention_mask, label = batch
        logits = tweet_model(input_ids=token_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask
                             )
