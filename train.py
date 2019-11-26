import torch

from util import TweetDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models import BertHashtag


if __name__ == '__main__':
    warm_start = False
    PATH = './checkpoints/'

    device = torch.device('cuda:0')
    n_epoch = 50

    train_data = TweetDataset(file_path='./data/train.txt',
                              meta_path='./data/meta.txt',
                              max_len=64)

    tweet_model = BertHashtag()
    tweet_model = tweet_model.to(device)

    if warm_start:
        tweet_model.load_state_dict(torch.load(PATH)['model_state_dict'])

    tweet_model.train()

    loss_func = CrossEntropyLoss(reduction='mean')

    optimizer = Adam(tweet_model.parameters(), lr=1e-5)

    for epoch in range(n_epoch):
        cum_loss = 0
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
        print('Epoch {} starts!'.format(epoch+1))

        for it, (token_ids, token_type_ids, attn_mask, label) in enumerate(train_dataloader):
            optimizer.zero_grad()

            token_ids, token_type_ids, attn_mask = token_ids.to(device), token_type_ids.to(device), attn_mask.to(device)
            label = label.to(device)

            score = tweet_model(input_ids=token_ids,
                                token_type_ids=token_type_ids,
                                attn_mask=attn_mask
                                )

            loss = loss_func(score, label)
            loss.backward()
            cum_loss += loss.item()

            optimizer.step()

            if (it + 1) % 100 == 0:
                print('Avg {}-th iteration loss: {}'.format(it+1, cum_loss/100))
                cum_loss = 0

        save_name = './checkpoints/tweet_model_gpu_checkpoints_epoch_{}.tar'.format(epoch+1)

        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': tweet_model.state_dict(),
                        'loss': loss},
                       save_name
                       )
