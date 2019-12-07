import torch
import json
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from models import BertHashtag
from util import TweetDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    model_name = 'max_seq_12-batch_size_256-lr_0.0001-schedule_ExponentialLR-gamma_0.8-epoch_20'
    model_path = './checkpoints/checkpoints-' + model_name + '.tar'

    max_len = 16
    device = torch.device('cuda:0')

    bert = BertHashtag(num_class=3)
    bert = bert.to(device)

    bert.load_state_dict(torch.load(model_path)['model_state_dict'])

    dev = TweetDataset('./data/dev.txt', './data/meta.txt', max_len)
    dev = dev

    bert.eval()
    dev_dataloader = DataLoader(dev, batch_size=64, shuffle=False)

    y_true = []
    y_pred = []
    y_pred_prob = []

    for _, (token_ids, token_type_ids, attn_mask, label) in enumerate(dev_dataloader):
        token_ids, token_type_ids, attn_mask = token_ids.to(device), token_type_ids.to(device), attn_mask.to(device)

        out = bert(input_ids=token_ids,
                   token_type_ids=token_type_ids,
                   attn_mask=attn_mask
                   )

        v, k = torch.max(out, -1)

        y_true += label.numpy().tolist()
        y_pred += k.cpu().numpy().tolist()
        y_pred_prob += v.detach().cpu().numpy().tolist()

    with open(model_name + 'txt', 'w') as f:
        json.dump({'y_pred': y_pred, 'y_pred_prob': y_pred_prob}, f)

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    print('f1 score is: {}\naccuracy is: {}\nconfusion maxtrix is:'.
          format(round(f1, 3), round(acc, 3)))
    print(confusion_matrix)
