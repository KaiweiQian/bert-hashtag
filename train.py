import torch

from util import TweetDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from models import BertHashtag


if __name__ == '__main__':
    warm_start = False
    PATH = './checkpoints/'

    device = torch.device('cuda:0')

    n_epoch = 10
    max_len = 32
    batch_size = 128
    max_grad_norm = 100.0
    scheduler_name = 'ExponentialLR'

    train_data = TweetDataset(file_path='./data/train.txt',
                              meta_path='./data/meta.txt',
                              max_len=max_len)

    tweet_model = BertHashtag(num_class=3, fix_bert=False)
    tweet_model = tweet_model.to(device)

    if warm_start:
        tweet_model.load_state_dict(torch.load(PATH)['model_state_dict'])

    tweet_model.train()

    loss_func = CrossEntropyLoss(reduction='mean')

    lr = 1e-3

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(tweet_model.parameters(), lr=lr, eps=1e-10)
    scheduler = ExponentialLR(optimizer, gamma=1/2)

    for epoch in range(n_epoch):
        cum_loss = 0
        cum_acc = 0

        print('Epoch {} starts!'.format(epoch+1))

        for it, (token_ids, token_type_ids, attn_mask, label) in enumerate(train_dataloader):
            token_ids, token_type_ids, attn_mask = token_ids.to(device), token_type_ids.to(device), attn_mask.to(device)
            label = label.to(device)

            score = tweet_model(input_ids=token_ids,
                                token_type_ids=token_type_ids,
                                attn_mask=attn_mask
                                )

            loss = loss_func(score, label)
            loss.backward()

            cum_loss += loss.item()
            cum_acc += torch.mean((torch.argmax(score, dim=1) == label).float())

            clip_grad_norm_(tweet_model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # if (it + 1) % 50 == 0:
            #    print('Avg {}-th iteration loss: {} and accuracy: {}'.
            #          format(it+1, cum_loss/50, cum_acc/50))
            #    cum_loss = 0
            #    cum_acc = 0

        print('{}-th epoch loss: {} and accuracy: {}'.
              format(epoch+1, cum_loss/len(train_dataloader), cum_acc/len(train_dataloader)))

        scheduler.step()

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        if (epoch + 1) % 5 == 0:
            save_name = './checkpoints/checkpoints-max_seq_{}-batch_size_{}-lr_{}-schedule_{}-epoch_{}.tar'. \
                format(max_len, batch_size, lr, scheduler_name, epoch + 1)

            torch.save({'epoch': epoch + 1,
                        'model_state_dict': tweet_model.state_dict(),
                        'loss': loss},
                       save_name
                       )
