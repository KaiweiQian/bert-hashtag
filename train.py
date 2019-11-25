import torch

from util import TweetDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from models import BertHashtag


if __name__ == '__main__':
    device = torch.device('cuda:0')
    n_epoch = 20

    train_data = TweetDataset(file_path='./data/train.txt',
                              meta_path='./data/meta.txt',
                              max_len=256)

    dev_data = TweetDataset(file_path='./data/dev.txt',
                            meta_path='./data/meta.txt',
                            max_len=256)

    tweet_model = BertHashtag.from_pretrained('bert-base-uncased')
    tweet_model = tweet_model.to(device)
    tweet_model.train()

    loss_func = CrossEntropyLoss(reduction='mean')

    optimizer = Adam(tweet_model.parameters(), lr=1e-5)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

    for epoch in range(n_epoch):
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

            optimizer.step()

            if (it + 1) % 100 == 0:
                print('{}-th iteration loss: {}'.format(it+1, loss.item()))

        save_name = './checkpoints/tweet_model_gpu_checkpoints_epoch_{}.tar'.format(epoch+1)

        torch.save({'epoch': epoch+1,
                    'model_state_dict': tweet_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   save_name
                   )
