import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from models import BertHashtag
from util import TweetDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    model_path = './checkpoints/tweet_3_class_gpu_checkpoints_lr_1e-07_epoch_20.tar'
    max_len = 32
    device = torch.device('cuda:0')

    bert = BertHashtag(num_class=3)
    bert = bert.to(device)

    bert.load_state_dict(torch.load(model_path)['model_state_dict'])

    dev = TweetDataset('./data/dev.txt', './data/meta.txt', max_len)
    dev = dev.to(device)

    bert.eval()
    dev_dataloader = DataLoader(dev, batch_size=512, shuffle=False)

    y_true = []
    y_pred = []
    y_pred_prob = []

    for _, (token_ids, token_type_ids, attn_mask, label) in enumerate(dev_dataloader):
        out = bert(input_ids=token_ids,
                   token_type_ids=token_type_ids,
                   attn_mask=attn_mask
                   )
        v, k = torch.max(out, -1)

        y_true += label.numpy().cpu().tolist()
        y_pred += k.numpy().cpu().tolist()
        y_pred_prob += v.detach().numpy().cpu().tolist()

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    print('f1 score is: {}\naccuracy is: {}\nconfusion maxtrix is:'.format(f1, acc))
    print(confusion_matrix)
