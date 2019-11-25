import torch

from util import TweetDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models import BertHashtag


if __name__ == '__main__':
    device = torch.device('cuda:0')

    train_data = TweetDataset(file_path='./data/train.txt',
                              meta_path='./data/meta.txt',
                              max_len=256)

    dev_data = TweetDataset(file_path='./data/dev.txt',
                            meta_path='./data/meta.txt',
                            max_len=256)

    tweet_model = BertHashtag.from_pretrained('bert-base-uncased')
    loss_func = CrossEntropyLoss(reduction='mean')

    optimizer = Adam(tweet_model.parameters(), lr=5e-4)

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    for it, (token_ids, token_type_ids, attn_mask, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        token_ids, token_type_ids, attn_mask = token_ids.to(device), token_type_ids.to(device), attn_mask.to(device)
        label = label.to(device)

        score = tweet_model(input_ids=token_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attn_mask
                            )

        loss = loss_func(score, label)
        loss.backward()
